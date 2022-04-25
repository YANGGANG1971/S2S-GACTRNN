import tensorflow.keras as keras
import tensorflow as tf
from GActrnn import GACTRNNCell
from Datasets_differlength import read_data, read_test_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import Prediction as pre

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

units_vec = [20,16,12,8]
t_vec = [1,6,36,216]
num = 100
step = 1
delay = 10
EPOCHS = 1000
val_split = 0.15
BATCH_SIZE = 32
patience = 20

# encoder
encoder_inputs = keras.layers.Input(shape=(num, 1), name='encoder_input')
# Create a list of RNN Cells, these are then concatenated into a single layer with the RNN layer.
encoder_cells = []
encoder_cells.append(GACTRNNCell(units_vec=units_vec, t_vec=t_vec, connectivity='dense'))
encoder = keras.layers.RNN(encoder_cells, return_state=True)
encoder_outputs, encoder_states = encoder(encoder_inputs)

# decoder
decoder_cells = []
decoder_cells.append(GACTRNNCell(units_vec=units_vec, t_vec=t_vec, connectivity='dense'))
decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
# last encoder output shape: (batch_size, 1, hidden_dim)
last_encoder_output = tf.expand_dims(encoder_outputs, axis=1)
# replicated last encoder output shape: (batch_size, window_size_future, hidden_dim)
replicated_last_encoder_output = tf.repeat(
    input=last_encoder_output,
    repeats=delay,
    axis=1)
# Set the initial state of the decoder to be the ouput state of the encoder.
decoder_outputs, decoder_states = decoder(replicated_last_encoder_output,
                                          initial_state=[encoder_states])
# Apply a dense layer with linear activation to set output to correct dimension.
decoder_dense = keras.layers.Dense(1)
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.models.Model(inputs=encoder_inputs, outputs=decoder_outputs)
model.summary()
model.load_weights('s2s_model_weight.hdf5')
model.summary()

# testing
x_train, y_train, scaler = read_data(num, step, delay)
test_data = read_test_data()
trans_data = scaler.transform(test_data.reshape(-1,1)).reshape(1,-1,1)

# reactive loop
print("**********Reactiveloop Time**********")
predict_open = pre.OpenLoop(trans_data, model, num, delay)
predict_open = scaler.inverse_transform(predict_open.reshape(-1,1)).flatten()

# proactive loop
print("**********Proactiveloop Time**********")
inputs = trans_data[:, :num, :]
loops = (test_data.shape[1] - num)//delay
predict_closed = pre.CloseLoop(inputs, model, delay, loops)
predict_closed = scaler.inverse_transform(predict_closed.reshape(-1,1)).flatten()

# plot figures
data = test_data.flatten()
fig1 = plt.figure()
ax1=fig1.add_subplot(111)
ax1.plot(data[num:], linestyle="--")
ax1.plot(predict_open, linestyle="-")
ax1.set_xlabel('Time step')
ax1.set_ylabel('Amplitude')
ax1.set_title('Reactive Loop')
plt.legend(["Ground truth",'predict'])
plt.show()

fig2 = plt.figure()
ax2=fig2.add_subplot(111)
ax2.plot(data[num:], linestyle="--")
ax2.plot(predict_closed, linestyle="-")
ax2.set_xlabel('Time step')
ax2.set_ylabel('Amplitude')
ax2.set_title('Proactive Loop')
plt.legend(["Ground truth",'predict'])
plt.show()

fig2 = plt.figure()
ax2=fig2.add_subplot(111)
ax2.plot(predict_open-data[num:], linestyle="-")
ax2.set_xlabel('Time step')
ax2.set_ylabel('Amplitude')
ax2.set_title('Error')
# ax2.set_yticks([-2.5,-2.0,-1.5,-1.0,-1,-0.5,0,0.5,1,1.5,2.0,2.5])
ax2.set_yticks([-8,-6,-4,-2,0,2,4,6,8])
plt.show()