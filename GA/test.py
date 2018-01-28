import time
import numpy as np


show_length = 40
num_iteration = 1000
base_char = '=='
show_char = '->'
threshold = 0
for i in range(num_iteration):
    temp = [base_char for _ in range(show_length)]
    threshold += + np.random.randint(0, 3)
    threshold %= show_length
    if threshold >= show_length:
        threshold = show_length - 1
    for j in range(threshold):
        temp[j] = '||'
    temp[threshold] = show_char
    rate = (threshold / show_length) * 100
    print(('\r' + ''.join(str(_item) for _item in temp) + "\t %.2f") % rate + "%", end='')
    # print(''.join(str(_item) for _item in temp))
    time.sleep(np.random.uniform(0, 1))
