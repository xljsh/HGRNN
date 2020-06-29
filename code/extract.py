import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'model/default/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/default', 'checkpoint path')
tf.flags.DEFINE_string('start_string', 'I am a boy,', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 1000, 'max length to generate')
tf.flags.DEFINE_string('embed_text', 'data/embed.txt', 'text with embed information')


def main(_):
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)
    # rb \r\n will not be converted into \n by system
    with open(FLAGS.embed_text, 'r') as f:
        text = f.read()
    start = converter.text_to_arr(FLAGS.start_string)
    # the beginning string must be same
    # arr = model.extract(text, start, converter.vocab_size)
    arr = model.extract_step_10(text, start, converter.vocab_size)
    # arr = model.extract_2_bit(text, start, converter.vocab_size)

    print(arr)


if __name__ == '__main__':
    tf.app.run()
