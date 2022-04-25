//***This is a description of S2S-GACTRNN***//

1. The Keras machine learning framework is used. The emvironment requirements are listed in the attachment file.

2. Use "main.py" to train a model, or directly use "test.py" to obtain the results. Before you use "test.py", ensure that you put the coresponding data sets into the folder "train_set" and "test_set".

3. By using "GACTRNNCell" to construct encoder-decoder model, the S2S-GACTRNN can be constructed. You can change "units_vec, t_vec, num and delay" to change the number of neurons, timescales, input length and output length, respectively. Additionally, these modules can be interconnected recurrently using different connectivity strategies ("dense, adjacent, etc.").

4. The encoder inputs are the input sequences, and the decoder inputs are the encoder outputs. You can use "model.summary()" to see the structure of the constructed model. A dense layer with linear activation is applied to set output to correct dimension.

5. In "model.compile()" you can choose optimizer and loss function before training the model by "model.fit()".

6. To realize the "reactive loop" and "proactive loop" please see "Prediction.py".

7. You can use "model.save_weights('s2s_model_weight.hdf5')" to save the parameters of a trained model, and use "model.load_weights('s2s_model_weight.hdf5')" to load these parameters for testing if necessary.

