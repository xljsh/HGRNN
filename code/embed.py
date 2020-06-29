import os
import collections
import time
import re
import random
import nltk
import pickle
import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN

# nltk.download('punkt')

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 1, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', '../model/default/converter.pkl', '../model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', '../model/default', 'checkpoint path')
tf.flags.DEFINE_string('start_string', 'I am a boy,', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 1000, 'max length to generate')


def generate_secrete_info_random(file="../data/anna.txt", info_len=8):
    with open(file, "r") as f:
        text = f.read()
    index = random.randint(0, len(text) - info_len)
    return text[index:index+8]


def get_start_string(file="../data/anna.txt", num=100, start_len=3):
    with open(file, "r") as f:
        text = f.read()
    text = nltk.sent_tokenize(text)
    start_strings = []
    i = 0
    index = random.randint(0, len(text) - 10 * num)
    while i < num:
        if '\n' not in text[index]:
            start_str = ' '.join(text[index].split()[:start_len])
            if len(start_str) >= 10:
                start_strings.append(start_str)
                i += 1
        index += 1
    # print(start_strings)
    # exit(0)
    return start_strings


def get_num_of_word(text):
    return len(re.compile(r'\w+').findall(text))


def perplexity(text, dict_word):
    testset = re.compile(r'\w+').findall(text)
    ppl = 1
    N = 0
    for word in testset:
        if word not in dict_word.keys():
            print(word)
            continue
        N += 1
        ppl = ppl * (1 / dict_word[word])
    ppl = pow(ppl, 1 / float(N))
    return ppl


def count_word(tokens):
    word_count = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        if f in word_count.keys():
            word_count[f] += 1
        else:
            word_count[f] = 1
    for word in word_count:
        word_count[word] = word_count[word] / float(sum(word_count.values()))
    return word_count


def main(_):
    batch_generate = False
    test_embed_text_path = "../data/embed.txt"
    embed_text_save_path = "../data/generate/"

    converter = TextConverter(filename=FLAGS.converter_path)
    # print(converter.vocab_size)
    # print(converter.vocab)
    # print(converter.text_to_arr("To whom is"))

    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)
    # print(converter.vocab_size)
    # print(FLAGS.checkpoint_path)
    # print(model.num_seqs, model.num_steps)
    model.load(FLAGS.checkpoint_path)

    # corpus = open(train_text_path, 'r').read()
    # corpus = gutenberg.words()
    with open("../model/count.pkl", 'rb') as f:
        word_count = pickle.load(f)

    if not batch_generate:
        start = converter.text_to_arr(FLAGS.start_string)
        arr = model.embed("changsha", start, converter.vocab_size)
        arr = converter.arr_to_text(arr)

        # w mode: \n is automatically converted to \n\r by system
        with open(test_embed_text_path, 'w') as f:
            f.write(arr)

        ppl_val = perplexity(arr, word_count)
        print(ppl_val)
    else:
        num_generate_text = 100
        start_str_num = 10
        start_str_word_num = 3
        for i in range(num_generate_text):
            start_str_list = get_start_string(num=start_str_num, start_len=start_str_word_num)
            secrete_info = generate_secrete_info_random()
            best_text = ''
            best_ppl = 1e10  # set a max value
            best_text_info = []
            for index, start_str in enumerate(start_str_list):
                file_path = embed_text_save_path + str(start_str_word_num) + "_" + str(len(secrete_info)) \
                            + "_" + str(i) + ".txt"
                # print(secrete_info, start_str)
                start_time = time.time()
                text = model.embed_step_10(secrete_info, converter.text_to_arr(start_str), converter.vocab_size)
                end_time = time.time()

                text = converter.arr_to_text(text)
                # computer the value of perplexity
                ppl_val = perplexity(text, word_count)
                print(ppl_val)
                # write the steganography text into txt file in batch
                with open(file_path, 'a+') as f:
                    # f.seek(0)
                    # f.truncate()
                    f.write("The generated text's number: %d; the number of words: %d; the number of characters: %d; "
                            "the bit length of secret message: %d; the embedding rate: %.2lf;"
                            " time of generating the text: %.2f sec; the perplexity of the text: %.2f\n"
                            % (index, get_num_of_word(text), len(text), len(secrete_info)*7,
                               len(secrete_info)/len(text), end_time-start_time, ppl_val))
                    f.write(text+"\n")
                if ppl_val < best_ppl:
                    best_ppl = ppl_val
                    best_text = text
                    best_text_info = [index, get_num_of_word(text), len(text), len(secrete_info)*7,
                                      len(secrete_info)/len(text), end_time-start_time, ppl_val]
            with open(file_path, 'a+') as f:
                f.write("The best generated text's number: %d; the number of words: %d; the number of characters: %d; "
                        "the bit length of secret message: %d; the embedding rate: %.2lf;"
                        " time of generating the text: %.2f sec; the best perplexity of the text: %.2f\n"
                        % (best_text_info[0], best_text_info[1], best_text_info[2], best_text_info[3],
                           best_text_info[4], best_text_info[5], best_text_info[6]))
                f.write(best_text+'\n')


if __name__ == '__main__':
    tf.app.run()
