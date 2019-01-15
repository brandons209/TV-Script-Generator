#general imports
import helper
import numpy as np
import time
import argparse

#pre-processing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#model
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.layers import Embedding, LSTM

#training
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import optimizers as opt

#argparser
parser = argparse.ArgumentParser()
parser.add_argument("-text_path", type=str, default="data/moes_tavern_lines.txt", help="path to a single text file or a directory contain text files")
parser.add_argument("-seq_length", type=int, default=15, help="sequence length of input into the network")
parser.add_argument("-continue", type=str, default=None, help="path to weights file if continuing training.")
options = parser.parse_args()

#load script text, replace with another file if you want.
script_text = helper.load_script(options.text_path)

#hyperparameters
learn_rate = 0.001
optimizer = opt.Adam(lr=learn_rate)
sequence_length = int(options.seq_length)
epochs = 200
batch_size = 32

#print some stats about the data
print('----------Dataset Stats-----------')
print('Approximate number of unique words: {}'.format(len({word: None for word in script_text.split()})))
scenes = script_text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {:.0f}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {:.0f}'.format(np.average(word_count_sentence)))

#tokenize inputs, we will be converting all non-word characters to words, like ! will be ||explaination||, characters not converted are left there
#turn all characters to lowercase to reduce vocab and train faster
tokens = Tokenizer(filters='', lower=True, char_level=False)
script_text = helper.tokenize_punctuation(script_text) #helper function to convert non-word characters
script_text = script_text.lower()

script_text = script_text.split()

tokens.fit_on_texts(script_text)

word_to_int = tokens.word_index #grab word : int dict
int_to_word = {int: word for word, int in word_to_int.items()} #flip word_to_int dict to get int to word
int_script_text = np.array([word_to_int[word] for word in script_text], dtype=np.int32) #convert text to integers
int_script_text, targets = helper.gen_sequences(int_script_text, sequence_length) #transform int_script_text to sequences of sequence_length and generate targets
vocab_length = len(word_to_int) + 1 #vocab_length for embedding needs to 1 one to length of the dictionary.

print("Number of vocabulary: {}".format(vocab_length))

#save dictionaries for use with testing model, also need to save sequence length since it needs to be the same when running test_model.py
start_time = time.strftime("%a_%b_%d_%Y_%H:%M", time.localtime())
save_path = "data/dicts/" + start_time + "/"

if options.continue is None:
    helper.save_dict(word_to_int, save_path, 'word_to_int.pkl')
    helper.save_dict(int_to_word, save_path, 'int_to_word.pkl')
    helper.save_dict(sequence_length, save_path, 'sequence_length.pkl')

#model definition
model = Sequential()
"""
input length = sequence_length since embedding requires specification of the length of each input.
if we were to convert our data into sentences instead, it would be a input length of the length of each sentence, leaving it unset caused NAN weights because i think each word just kept added to a super continous long vector?
can also do two words in a sequence, then generate two words, or two words in and one word out.
also split up text so that each line is one sequence, and pad/truncate sentences to a length, and train on that. this can allow the network to utilize LSTM cells to learn context, pretty easy to change how the data is split doing that.
"""
model.add(Embedding(vocab_length, 300, input_length=sequence_length))
model.add(LSTM(400, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(400, dropout=0.3, recurrent_dropout=0.3))
#model.add(Dropout(0.2))
model.add(Dense(vocab_length, activation='softmax'))

#compile model using adam optmizer, this loss function calculates categorical_crossentropy without having to one-hot-encode the labels, this gives much better computational times since we are not doing matrix multiplication on huge matrixies of zeros. espically with a large vocab.
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
print("\n")

print("Press enter to start training with these hyperparameters:")
print("Learn rate: {}, Sequence Length: {}, Epochs: {}, Batch Size: {}".format(learn_rate, sequence_length, epochs, batch_size))
input("\n")
#training:
#view tensorboard with command: tensorboard --logdir=tensorboard_logs
ten_board = TensorBoard(log_dir='tensorboard_logs/{}'.format(start_time), write_images=True)
weight_save_path = 'saved_weights/model.best.weights.hdf5'
checkpointer = ModelCheckpoint(weight_save_path, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)
print("Tensorboard logs for this run are in: {}, weights will be saved in {}\n".format('tensorboard_logs/{}'.format(start_time), weight_save_path))
if options.contine is not None:
    model.load_weights(options.continue)
model.fit(int_script_text, targets, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, ten_board, early_stop])

#load best weights and saved model
print("Loading best weights and saving model in {}".format('saved_models/model.{}.h5'.format(start_time)))
model.load_weights(weight_save_path)
model.save('saved_models/model.{}.h5'.format(start_time))

#generate new script
def generate_text(seed_text, num_words):
    input_text= seed_text
    for _  in range(num_words):
        #tokenize text to ints
        int_text = helper.tokenize_punctuation(input_text)
        int_text = int_text.lower()
        int_text = int_text.split()
        int_text = np.array([word_to_int[word] for word in int_text], dtype=np.int32)
        #pad text if it is too short, pads with zeros at beginning of text, so shouldnt have too much noise added
        int_text = pad_sequences([int_text], maxlen=sequence_length)
        #predict next word:
        prediction = model.predict(int_text, verbose=0)
        output_word = int_to_word[helper.sample(prediction, temp=0.3)]
        #append to the result
        input_text += ' ' + output_word
    #convert tokenized punctuation and other characters back
    result = helper.untokenize_punctuation(input_text)
    return result

seed = input("Enter seed to generate text.\n")

if seed != "":
    print(generate_text(seed, 200))
