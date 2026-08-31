#!/usr/bin/env python3

import os
import time
import subprocess

if __name__ == "__main__":
    root = os.getcwd()
    choices = input(
            "Enter specific dirs or sub-dirs (comma-separated) or press Enter to scan all: "
            ).strip()
    selected_dirs = choice.split(',') if choices else None

    print("Test Started.")
    cnt = 0
    err_list = []
    test_start_time = time.time()
    for root, dirs, files in sorted(os.walk(root), key=lambda strname: strname[0]):
        if selected_dirs and all(root != dirname for dirname in selected_dirs):
            continue
        for file in sorted(files):
            if file.endswith(".py") and \
                    not (root == os.getcwd() and file == os.path.basename(__file__)): 
                    # Avoid running itself 
                cnt += 1
                file_path = os.path.join(root, file)
                print("No.{0:n}\t: {1}".format(cnt, file_path))
                try:
                    subprocess.run(["python", file_path],\
                            stdout = subprocess.DEVNULL,\
                            # stderr = subprocess.DEVNULL,\
                            check=True)
                except subprocess.CalledProcessError as err:
                    err_list.append(file_path)
    test_end_time = time.time()
    time_elapse = test_end_time - test_start_time
    print("Total Files: {0:n}; Total Errors: {1:n} ".format(cnt, len(err_list)))
    for file_path in err_list:
        print(file_path)
    print("Total Wall Time Spent: {0:.3f} s.".format(time_elapse))
    print("Test Ended.")

