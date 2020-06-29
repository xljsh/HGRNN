import tensorflow as tf

inputs = tf.placeholder(tf.float32, shape=(2, 3, 1), name='inputs')
cell = tf.nn.rnn_cell.BasicLSTMCell(256)
initial_state = cell.zero_state(2, tf.float32)
out, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
