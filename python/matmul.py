import numpy as np
import json
import time
from sklearn.metrics import mean_squared_error


with open('../conf/settings.json') as f:
    settings = json.load(f)

path = settings['path']
iter = settings['iterations']

for size in settings['sizes']:
    A = np.loadtxt(path + "matA_" + str(size['N']) + ".dat", skiprows=1, dtype='float')
    B = np.loadtxt(path + "matB_" + str(size['N']) + ".dat", skiprows=1, dtype='float')
    C = np.loadtxt(path + "matC_serial_" + str(size['N']) + ".dat", skiprows=1, dtype='float')

    A = A.reshape(size['N'], size['N'])
    B = B.reshape(size['N'], size['N'])

    start = time.time_ns()
    D = np.dot(A, B)
    end = time.time_ns()

    timeT = end - start
    error = mean_squared_error(C.reshape(1, size['N']*size['N']), D.reshape(1, size['N']*size['N']))
    print(timeT/1000)
    print(error)

f.close()
