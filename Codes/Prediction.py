import numpy as np
import time

def CloseLoop(inputs, model, delay, loops):
    '''
    make a close-loop prediction.
    '''
    if len(np.shape(inputs)) != 3:
        raise ValueError('Written by Lintor: Inputs should have 3 dims.')
    shape = np.shape(inputs)
    if shape[0] != 1:
        raise ValueError('Written by Lintor: Inputs dims should be like [1,...,...].')

    for n in range(loops):
        pred = model.predict(inputs)
        print(time.time())
        inputs = inputs.reshape(shape[1], shape[2])
        delete = list(range(delay))
        inputs = np.delete(inputs, delete, axis=0)
        inputs = np.row_stack((inputs, pred[0, -delay:, :]))
        inputs = inputs.reshape(shape[0], shape[1], shape[2])
        if n ==0:
            outputs = inputs[0, -delay:, :]
        else:
            outputs = np.concatenate((outputs, inputs[0, -delay:, :]))
    return np.array(outputs)

def OpenLoop(inputs, model, num, delay):
    '''
    make a open-loop prediction.
    '''
    if len(np.shape(inputs)) != 3:
        raise ValueError('Written by Lintor: Inputs should have 3 dims.')

    shape = np.shape(inputs)
    outputs = []

    loops = (shape[1] - num) // delay
    if loops == 0:
        raise ValueError('Written by Lintor: Larger second dims of inputs.')

    for n in range(loops):
        items = inputs[:, n*delay: (num+n*delay), :]
        pred = model.predict(items)
        print(time.time())
        if n ==0:
            outputs = pred[0, -delay:, :]
        else:
            outputs = np.concatenate((outputs, pred[0, -delay:, :]))
    return np.array(outputs)

if __name__ == '__main__':
    x = np.random.random([1,10,2])
    shape = np.shape(x)
    # y = OneStep(x, None, 2, 3)