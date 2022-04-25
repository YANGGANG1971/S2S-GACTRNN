import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def read_data(num, step, delay):
    data = []
    input_path = "./train_set"
    files = os.listdir(input_path)

    for file in files:
        if 'xlsx' in file:
            filepath = os.path.join(input_path, file)
            df = pd.read_excel(filepath, header=None).values
            data.append(df[:, 2])

    all_data = data[0]
    for i in range(1, len(data)):
        all_data = np.concatenate((all_data, data[i]))
    scaler = StandardScaler()
    scaler.fit(all_data.reshape((-1, 1)))
    for i in range(len(data)):
        data[i] = scaler.transform(data[i].reshape(-1,1)).flatten()

    x = []
    y = []
    for k in range(len(data)):
        item = data[k]
        limit = (item.shape[0] - delay - num) // step
        for i in range(limit):
            x.append(np.array([item[step * i: step * i + num]]).T)
            y.append(np.array([item[step * i + num: step * i + num + delay]]).T)

    return np.array(x), np.array(y), scaler

def read_test_data():
    data = []
    input_path = "./test_set"
    files = os.listdir(input_path)

    for file in files:
        if 'xlsx' in file:
            filepath = os.path.join(input_path, file)
            df = pd.read_excel(filepath, header=None).values
            data.append(df[:, 2])
    data = data[0].reshape((1, -1, 1))
    return data


