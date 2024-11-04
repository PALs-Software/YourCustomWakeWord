import sys
sys.path.insert(0, "./")
from modules.logging import get_last_tested_setting_no_from_log
import subprocess
import time

print('Start train model configuration test:')

while True:
    last_test_no = get_last_tested_setting_no_from_log()
    if last_test_no > 1:
        last_test_no -= 1

    subprocess.run(["python", "3_train_model/0_test_train_model_configurations.py", f'--StartBySettingNo {last_test_no}'])

    print("Process got killed, restart test script, after waiting for 30 seconds") # Restart the process when its got killed by the os, because of to much memory consuption (Currently something in tensorflow is buggy and leaking memory with the time)
    time.sleep(30)