# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import re
from read_utils import TextConverter


def pick_top_n(preds, vocab_size, top_n=2):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    c = np.argsort(p)[vocab_size-top_n:]
    return c


def pick_top_n_random(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)
    # return [c, tmp[c]]
    return c


def encode(s):
    return ' '.join([bin(ord(c)).replace('0b', '') for c in s])


def decode(s):
    return ''.join([chr(i) for i in [int(b, 2) for b in s.split(' ')]])


def int2bin(n, count=2):
    return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def get_string_bit(string):
    message = [s if len(s) == 7 else s + '0' for s in encode(string).split(' ')]
    message = ''.join(message).replace(' ', '')
    message_length = len(message)
    message_length = bin(message_length).replace('0b', '')
    message_length = message_length if len(message_length) == 7 else '0' * (
            7 - len(message_length)) + message_length
    return message_length + message


class CharRNN:
    def __init__(self, num_classes, num_seqs=32, num_steps=50,
                 lstm_size=128, num_layers=1, learning_rate=0.002,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        # self.lstm_inputs = None
        #
        # self.lstm_outputs_C = None
        # self.lstm_outputs_S = None
        # self.lstm_outputs_W = None
        # self.lstm_outputs_P = None
        #
        # self.initial_state_C = None
        # self.initial_state_S = None
        # self.initial_state_W = None
        # self.initial_state_P = None
        # self.final_state_C = None
        # self.final_state_S = None
        # self.final_state_W = None
        # self.final_state_P = None
        #
        # self.final_input = None
        # self.g_t_cs = None
        # self.g_t_sw = None

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
                print(self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
            print(self.lstm_inputs.shape)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.variable_scope('lstm-c'):
            # LSTM-C
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state_C = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            # output --> ht; state-->(ct,ht)
            self.lstm_outputs_C, self.final_state_C = tf.nn.dynamic_rnn(cell, self.lstm_inputs,
                                                                        initial_state=self.initial_state_C)
            # print("LSTM-C")
            # print(self.lstm_inputs.get_shape(), tf.shape(self.initial_state_C), tf.shape(self.final_state_C))

        with tf.variable_scope('lstm-s'):
            self.weight_g = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_seqs], stddev=0.1))
            self.bias_g = tf.Variable(tf.zeros(self.num_seqs))
            # LSTM-S
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state_S = cell.zero_state(self.num_seqs, tf.float32)

            # gate
            # print("LSTM-S")
            # print(tf.shape(self.final_state_C), tf.shape(self.initial_state_S))
            v_t_s = tf.concat([self.final_state_C, self.initial_state_S], 0)
            # print(v_t_s.shape)
            self.g_t_cs = tf.tanh(tf.matmul(v_t_s, self.weight_g) + self.bias_g)
            # print(self.g_t_cs.shape)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs_S, self.final_state_S = tf.nn.dynamic_rnn(cell, self.lstm_outputs_C,
                                                                        initial_state=self.initial_state_S)
            self.final_state_S = tf.add(tf.matmul(self.g_t_cs, self.final_state_S),
                                        tf.matmul(1-self.g_t_cs, self.final_state_S))

        with tf.variable_scope('lstm-w'):
            self.weight_g = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_seqs], stddev=0.1))
            self.bias_g = tf.Variable(tf.zeros(self.num_seqs))
            # LSTM-W
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state_W = cell.zero_state(self.num_seqs, tf.float32)

            # gate
            v_t_w = tf.concat([self.final_state_S, self.initial_state_W], 0)
            self.g_t_sw = tf.tanh(tf.matmul(v_t_w, self.weight_g) + self.bias_g)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs_W, self.final_state_W = tf.nn.dynamic_rnn(cell, self.lstm_outputs_S,
                                                                        initial_state=self.initial_state_W)
            self.final_state_W = tf.add(tf.matmul(self.g_t_cs, self.final_state_W),
                                        tf.matmul(1-self.g_t_cs, self.final_state_W))

        with tf.variable_scope('lstm-p'):
            # LSTM-P
            # self.final_input = tf.concat([self.lstm_outputs_C, self.lstm_outputs_S, self.lstm_outputs_W], 1)
            self.final_input = tf.add(tf.add(self.lstm_outputs_C, self.lstm_outputs_S), self.lstm_outputs_W)
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state_P = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs_P, self.final_state_P = tf.nn.dynamic_rnn(cell, self.final_input,
                                                                        initial_state=self.initial_state_P)
            # print("p-out:", tf.shape(self.lstm_outputs_P))

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs_P, 1)
            # print(seq_output.get_shape())
            x = tf.reshape(seq_output, [-1, self.lstm_size])
            # print("x", tf.shape(x))

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            print(tf.shape(y_one_hot), self.logits.get_shape())
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state_C)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state_C: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state_P,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if step % save_every_n == 0:
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def load(self, checkpoint):
        self.session = tf.Session()
        # self.saver.restore(self.session, checkpoint)
        module_file = tf.train.latest_checkpoint(checkpoint)
        self.session.run(tf.global_variables_initializer())
        if module_file is not None:
            self.saver.restore(self.session, module_file)
        print('Restored from: {}'.format(checkpoint))

    def embed(self, string, prime, vocab_size):
        converter = TextConverter(filename='../model/default/converter.pkl')
        message = get_string_bit(string)

        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state_C)
        preds = np.ones((vocab_size,))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state_C: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state_P], feed_dict=feed)

        c = pick_top_n(preds, vocab_size)[0]
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        i = 0
        while True:
            if i >= len(message) and converter.arr_to_text([c]) in ['.', '!']:
                # print("formal", i, len(message), converter.arr_to_text([c]))
                break

            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state_C: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state_P], feed_dict=feed)
            print(np.shape(preds))
            if i < len(message):
                c = pick_top_n(preds, vocab_size)[int(message[i])]
                i += 1
            else:
                c = pick_top_n_random(preds, vocab_size)[0]
            samples.append(c)
        return np.array(samples)

    def extract(self, text, prime, vocab_size):
        converter = TextConverter(filename='../model/default/converter.pkl')
        text = text[len(prime)+1:]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size,))

        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)

        c = pick_top_n(preds, vocab_size)[0]
        length = ''
        for i in range(7):
            x = np.zeros((1, 1))
            # print(c)
            x[0, 0] = c
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)
            c = pick_top_n(preds, vocab_size)
            for index, item in enumerate(c):
                if converter.arr_to_text([item]) == text[i]:
                    length += str(index)
                    c = item
        text = text[7:]
        ex_string = ''
        for i in range(int("0b"+length, 2)):
            x = np.zeros((1, 1))
            try:
                x[0, 0] = c
            except ValueError:
                pass

            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            for index, item in enumerate(c):
                if converter.arr_to_text([item]) == text[i]:
                    ex_string += str(index)
                    c = item
        ex_string = re.findall(r'.{7}', ex_string)
        ex_string = ' '.join(ex_string).replace('1000000', '100000')
        return decode(ex_string)


if __name__ == '__main__':
    pass
    # model = CharRNN(81,
    #                 num_seqs=FLAGS.num_seqs,
    #                 num_steps=FLAGS.num_steps,
    #                 lstm_size=FLAGS.lstm_size,
    #                 num_layers=FLAGS.num_layers,
    #                 learning_rate=FLAGS.learning_rate,
    #                 train_keep_prob=FLAGS.train_keep_prob,
    #                 use_embedding=FLAGS.use_embedding,
    #                 embedding_size=FLAGS.embedding_size
    #                 )

