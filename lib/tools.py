# Copyright 2025 Prism Developers. All Rights Reserved.
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

import os
import psutil
import tempfile
import h5py
import numpy as np

# Memory management tools
def calculate_chunks(mr_adc, nmo_chunked, nmo_non_chunked, ntensors = 1, extra_tensors = [[]]):

    extra_mem = 0
    extra_tensors = np.asarray(extra_tensors)
    for nmo_extra_tensor in extra_tensors:
        extra_mem += np.prod(nmo_extra_tensor[nmo_extra_tensor > 0]) * 8/1e6

    avail_mem = (mr_adc.max_memory - extra_mem - current_memory()) * 0.9 / ntensors

    nmo_non_chunked = np.asarray(nmo_non_chunked)
    tensor_mem = np.prod(nmo_non_chunked[nmo_non_chunked > 0]) * 8/1e6

    chunk_size = int(avail_mem / tensor_mem)

    if chunk_size > nmo_chunked:
        chunk_size = nmo_chunked
    if chunk_size <= 0:
        chunk_size = 1

    mr_adc.log.debug("avail mem %.2f / %.2f MB, chunk size %.2f MB (%.2f MB) [current mem %.2f]", avail_mem, mr_adc.max_memory,
                                                                                                  chunk_size * tensor_mem,
                                                                                                  nmo_chunked * tensor_mem,
                                                                                                  current_memory())

    chunks_list = []
    for s_chunk in range(0, nmo_chunked, chunk_size):
        f_chunk = min(nmo_chunked, s_chunk + chunk_size)
        chunks_list.append([s_chunk, f_chunk])

    return chunks_list

def calculate_double_chunks(mr_adc, nmo_chunked, nmo_non_chunked_1, nmo_non_chunked_2, ntensors = 1, extra_tensors = [[]]):

    extra_mem = 0
    extra_tensors = np.asarray(extra_tensors)
    for nmo_extra_tensor in extra_tensors:
        extra_mem += np.prod(nmo_extra_tensor[nmo_extra_tensor > 0]) * 8/1e6

    avail_mem = (mr_adc.max_memory - extra_mem - current_memory()) * 0.9 / ntensors

    nmo_non_chunked_1 = np.asarray(nmo_non_chunked_1)
    nmo_non_chunked_2 = np.asarray(nmo_non_chunked_2)

    tensor_mem_1 = np.prod(nmo_non_chunked_1[nmo_non_chunked_1 > 0]) * 8/1e6
    tensor_mem_2 = np.prod(nmo_non_chunked_2[nmo_non_chunked_2 > 0]) * 8/1e6

    chunk_size = int(avail_mem / (tensor_mem_1 + tensor_mem_2))

    if chunk_size > nmo_chunked:
        chunk_size = nmo_chunked
    if chunk_size <= 0:
        chunk_size = 1

    mr_adc.log.debug("avail mem %.2f / %.2f MB, chunk size %.2f MB (%.2f MB) [current mem %.2f]", avail_mem, mr_adc.max_memory,
                                                                                                  chunk_size * (tensor_mem_1 + tensor_mem_2),
                                                                                                  nmo_chunked * (tensor_mem_1 + tensor_mem_2),
                                                                                                  current_memory())

    chunks_list = []
    for s_chunk in range(0, nmo_chunked, chunk_size):
        f_chunk = min(nmo_chunked, s_chunk + chunk_size)
        chunks_list.append([s_chunk, f_chunk])

    return chunks_list

def current_memory():

    pid = os.getpid()
    process = psutil.Process(pid)

    return process.memory_info().rss / 1024**2

# Disk managements tools
def create_temp_file(mr_adc, mode='r+', *args, **kwargs):

    temp_file = tempfile.NamedTemporaryFile(dir=mr_adc.temp_dir, delete=True)
    filename = temp_file.name

    return h5py.File(filename, mode, *args, **kwargs)

def create_dataset(dataset_name, temp_file, shape):

    if isinstance(temp_file, h5py.File):
        data = temp_file.create_dataset(dataset_name, shape, 'f8')
    else:
        data = np.zeros(shape)
    return data

def flush(temp_file):

    if isinstance(temp_file, h5py.File):
        temp_file.flush()
