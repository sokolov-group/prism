# Copyright 2026 Prism Developers. All Rights Reserved.
# Adapted from 2014-2018 PySCF
#
# Licensed under the GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
#
# See the License file for the specific language governing
# permissions and limitations.
#
# Available at https://github.com/sokolov-group/prism
#
# Authors: Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Logging system

Log level
---------

======= ======
Level   number
------- ------
DEBUG   6
EXTRA   5
INFO    4
NOTE    3
WARN    2
ERROR   1
QUIET   0
======= ======
'''

import sys
import time

if sys.version_info < (3, 0):
    process_clock = time.clock
    perf_counter = time.time
else:
    process_clock = time.process_time
    perf_counter = time.perf_counter

DEBUG  = 6
EXTRA  = 5
INFO   = 4
NOTE   = 3
WARN   = 2
ERROR  = 1
QUIET  = 0

DEBUG_LEVEL  = 6
TIMER_LEVEL  = 5
TIMER0_LEVEL = 4

import cProfile
import pstats
import io
from functools import wraps

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(15)
        print(s.getvalue())
        return result
    return wrapper

import psutil
import threading

###def detect_serial(func):
###    @wraps(func)
###    def wrapper(*args, **kwargs):
###        process = psutil.Process()
###        
###        cpu_before = psutil.cpu_percent(interval=None)
###        threads_before = process.num_threads()
###        
###        start = time.time()
###        result = func(*args, **kwargs)
###        duration = time.time() - start
###        
###        cpu_after = psutil.cpu_percent(interval=0.1)
###        threads_after = process.num_threads()
###        
###        avg_cpu = (cpu_before + cpu_after) / 2
###        
###        print(f"\n    {func.__name__}(): "
###              f"time = {duration:.3f}s, "
###              f"cpu = {avg_cpu:.2f}%, "
###              f"threads = {threads_after}\n")
###
###        # heuristic for serial: low CPU but long runtime
###        if duration > 0.1 and avg_cpu < 50:
###            print(f"WARN: Likely SERIAL section: {func.__name__}()")
###        return result
###    return wrapper


import statistics

def detect_serial(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        cpu_samples = []
        stop_event = threading.Event()
        peak_threads = [0]

        def sample_cpu(interval=0.05):
            # Warm up the cpu_percent call — first call always returns 0.0
            process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
            while not stop_event.is_set():
                # Per-process CPU across all its threads, normalized to 0–100%
                proc_cpu = process.cpu_percent(interval=None) / psutil.cpu_count()
                cpu_samples.append(proc_cpu)
                current_threads = process.num_threads()
                if current_threads > peak_threads[0]:
                    peak_threads[0] = current_threads
                time.sleep(interval)

        sampler = threading.Thread(target=sample_cpu, daemon=True)
        sampler.start()

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start
            stop_event.set()
            sampler.join(timeout=1.0)

        if not cpu_samples:
            cpu_samples = [0.0]

        avg_cpu = statistics.mean(cpu_samples)
        p95_cpu = statistics.quantiles(cpu_samples, n=20)[-1] if len(cpu_samples) >= 20 else max(cpu_samples)
        cpu_std = statistics.stdev(cpu_samples) if len(cpu_samples) > 1 else 0.0

        is_serial = (
            duration > 0.1        # long enough to be meaningful
            and avg_cpu < 50      # low average utilisation
            and p95_cpu < 70      # no sustained bursts either
            and peak_threads[0] <= process.num_threads()  # no extra threads spawned
        )

        status = "SERIAL" if is_serial else "parallel/async"
        print(
            f"\n  {func.__name__}()\n"
            f"    duration    : {duration:.3f}s\n"
            f"    cpu avg     : {avg_cpu:.1f}% (std={cpu_std:.1f}, p95={p95_cpu:.1f}%)\n"
            f"    samples     : {len(cpu_samples)}\n"
            f"    peak threads: {peak_threads[0]}\n"
            f"    verdict     : {status}\n"
        )
        if is_serial:
            print(f"  WARN: Likely SERIAL section detected in {func.__name__}()")
        return result
    return wrapper

import os

def detect_memory_pressure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        stop_event = threading.Event()

        # --- sampled metrics ---
        rss_samples   = []   # resident set size  — physical RAM in use
        vms_samples   = []   # virtual memory size — includes swapped pages
        swap_samples  = []   # system-wide swap in use
        page_fault_samples = []

        # baseline swap and page faults before execution
        sys_swap_before   = psutil.swap_memory().used
        mem_info          = process.memory_info()
        pfaults_before    = getattr(mem_info, 'num_page_faults',  # Windows
                            getattr(mem_info, 'majflt', 0))        # Linux

        def sample_memory(interval=0.05):
            while not stop_event.is_set():
                try:
                    mem        = process.memory_info()
                    swap       = psutil.swap_memory()
                    rss_samples.append(mem.rss)
                    vms_samples.append(mem.vms)
                    swap_samples.append(swap.used)

                    # minor + major faults accumulated since process start
                    faults = getattr(mem, 'num_page_faults',
                             getattr(mem, 'majflt', 0))
                    page_fault_samples.append(faults)
                except psutil.NoSuchProcess:
                    break
                time.sleep(interval)

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start
            stop_event.set()
            sampler.join(timeout=1.0)

        if not rss_samples:
            rss_samples = vms_samples = swap_samples = [0]

        def mb(val): return val / 1024 ** 2    # bytes → MB

        # --- RSS analysis ---
        rss_start     = mb(rss_samples[0])
        rss_peak      = mb(max(rss_samples))
        rss_end       = mb(rss_samples[-1])
        rss_growth    = rss_end - rss_start
        rss_delta_max = rss_peak - rss_start   # worst-case headroom needed

        # --- allocation rate: how fast was RSS growing on average ---
        alloc_rate    = rss_growth / duration if duration > 0 else 0

        # --- swap delta ---
        swap_start    = mb(swap_samples[0])
        swap_end      = mb(swap_samples[-1])
        swap_delta    = swap_end - swap_start

        # --- page faults delta ---
        pfaults_end   = page_fault_samples[-1] if page_fault_samples else 0
        pfaults_delta = pfaults_end - pfaults_before

        # --- implicit copy heuristic ---
        # RSS grew significantly faster than a single allocation would suggest,
        # meaning temporary copies likely existed alongside the original
        rss_std            = statistics.stdev(mb(s) for s in rss_samples) \
                             if len(rss_samples) > 1 else 0
        likely_copy        = rss_delta_max > rss_growth * 1.5 and rss_delta_max > 50

        # --- warnings ---
        warn_swap          = swap_delta    >  50    # MB
        warn_alloc_rate    = alloc_rate    > 500    # MB/s — matches ~2.8GB/s vmstat signal
        warn_page_faults   = pfaults_delta > 1000
        warn_copy          = likely_copy

        print(
            f"\n  {func.__name__}()\n"
            f"    duration      : {duration:.3f}s\n"
            f"    RSS start     : {rss_start:.1f} MB\n"
            f"    RSS peak      : {rss_peak:.1f} MB   (headroom needed: {rss_delta_max:.1f} MB)\n"
            f"    RSS end       : {rss_end:.1f} MB   (net growth: {rss_growth:+.1f} MB)\n"
            f"    alloc rate    : {alloc_rate:.1f} MB/s\n"
            f"    RSS std dev   : {rss_std:.1f} MB\n"
            f"    swap delta    : {swap_delta:+.1f} MB\n"
            f"    samples       : {len(rss_samples)}\n"
        )

        if warn_alloc_rate:
            print(f"  WARN [{func.__name__}]: High allocation rate "
                  f"({alloc_rate:.0f} MB/s) — possible implicit tensor copies")
        if warn_copy:
            print(f"  WARN [{func.__name__}]: Peak RSS {rss_delta_max:.0f} MB > "
                  f"1.5x net growth {rss_growth:.0f} MB — "
                  f"suggests temporary copies existed mid-execution")
        if warn_swap:
            print(f"  WARN [{func.__name__}]: Swap grew by {swap_delta:.0f} MB "
                  f"— RAM pressure during execution")
        if warn_page_faults:
            print(f"  WARN [{func.__name__}]: {pfaults_delta} page faults "
                  f"— kernel fetching pages from disk, swap debt likely")

        return result
    return wrapper

def flush(rec, msg, *args):
    rec.stdout.write(msg%args)
    rec.stdout.write('\n')
    rec.stdout.flush()

def log(rec, msg, *args):
    if rec.verbose > QUIET:
        flush(rec, msg, *args)

def error(rec, msg, *args):
    if rec.verbose >= ERROR:
        flush(rec, '\nERROR: '+msg+'\n', *args)
    sys.stderr.write('ERROR: ' + (msg%args) + '\n')

def warn(rec, msg, *args):
    if rec.verbose >= WARN:
        flush(rec, '\nWARN: '+msg+'\n', *args)
        if rec.stdout is not sys.stdout:
            sys.stderr.write('WARN: ' + (msg%args) + '\n')

def info(rec, msg, *args):
    if rec.verbose >= INFO:
        flush(rec, msg, *args)

def note(rec, msg, *args):
    if rec.verbose >= NOTE:
        flush(rec, msg, *args)

def extra(rec, msg, *args):
    if rec.verbose >= EXTRA:
        flush(rec, msg, *args)

def debug(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)

def stdout(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)
    sys.stdout.write('>>> %s\n' % msg)

def timer(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '> CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '> CPU time for %s %9.2f sec' % (msg, rec._t0-cpu0))
        return rec._t0

def timer0(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER0_LEVEL:
            flush(rec, '> CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= TIMER0_LEVEL:
            flush(rec, '> CPU time for %s %9.2f sec' % (msg, rec._t0-cpu0))
        return rec._t0

def timer_debug(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= DEBUG_LEVEL:
            flush(rec, '> CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= DEBUG_LEVEL:
            flush(rec, '> CPU time for %s %9.2f sec' % (msg, rec._t0-cpu0))
        return rec._t0

def prism_header(self):

    self.info("""\n
------------------------------------------------------------------------------

            PRISM: Open-Source implementation of ab initio methods
                    for excited states and spectroscopy

                               Version 0.9

                   Copyright (C) 2026 Prism Developers
                   Contributors: Nicholas Yiching Chiang
                                 Carlos E. V. de Moura
                                 Nishshanka M. Lakshan
                                 Rajat S. Majumder
                                 Ilia M. Mazin
                                 Donna H. Odhiambo
                                 Bryce Pickett
                                 James D. Serna
                                 Alexander Yu. Sokolov

            Unless required by applicable law or agreed to in
            writing, software distributed under the GNU General
            Public License v3.0 and is distributed on an "AS IS"
            BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
            either express or implied.

            See the License for the specific language governing
            permissions and limitations.

            Available at https://github.com/sokolov-group/prism

------------------------------------------------------------------------------\n""")

class Logger:
    '''
    Attributes:
        stdout : file object or sys.stdout
            The file to dump output message.
        verbose : int
            Large value means more noise in the output file.
    '''
    def __init__(self, stdout=sys.stdout, verbose=NOTE):
        self.stdout = stdout
        self.verbose = verbose
        self._t0 = process_clock()
        self._w0 = perf_counter()

    log = log
    error = error
    warn = warn
    info = info
    note = note
    extra = extra
    debug  = debug
    timer0 = timer0
    timer = timer
    timer_debug = timer_debug
    prism_header = prism_header

def new_logger(rec=None, verbose=None):
    '''Create and return a :class:`Logger` object

    Args:
        rec : An object which carries the attributes stdout and verbose

        verbose : a Logger object, or integer or None
            The verbose level. If verbose is a Logger object, the Logger
            object is returned. If verbose is not specified (None),
            rec.verbose will be used in the new Logger object.
    '''
    if isinstance(verbose, Logger):
        log = verbose
    elif isinstance(verbose, int):
        if getattr(rec, 'stdout', None):
            log = Logger(rec.stdout, verbose)
        else:
            log = Logger(sys.stdout, verbose)
    else:
        log = Logger(rec.stdout, rec.verbose)
    return log
