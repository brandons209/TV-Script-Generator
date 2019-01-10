from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys, os
import helper
import argparse

#load in model user specified
parser = argparse.ArgumentParser()
parser.add_argument("-model_path", type=str, help="path to saved model")
parser.add_argument("-dict_path", type=str, default="data/dicts/", help="directory path to dictionaries for converting words to tokens")
parser.add_argument("-temp", type=float, default="0.3", help="temperature value for choosing next word when generating, a value of 0 will always choose the word with the highest probability.")
options = parser.parse_args()

model = load_model(options.model_path)

#load dictionaries for converting text to ints and back
word_to_int = helper.load_dict(options.dict_path + 'word_to_int.pkl')
int_to_word = helper.load_dict(options.dict_path + 'int_to_word.pkl')
sequence_length = helper.load_dict(options.dict_path + 'sequence_length.pkl')

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
        output_word = int_to_word[helper.sample(prediction, temp=options.temp)]
        #append to the result
        input_text += ' ' + output_word
    #convert tokenized punctuation and other characters back
    result = helper.untokenize_punctuation(input_text)
    return result

#input amount of words to generate, and the seed text:
seed = input("Please enter the seed text to generate, good options are 'Homer_Simpson:', 'Bart_Simpson:', 'Moe_Szyslak:', or other character's names.\n")
num_words = input("Please enter amount of words to be generated as an integer.\n")
num_words = int(num_words)

#print amount of characters specified.
print("Starting seed is: {}\n\n".format(seed))
print(generate_text(seed, num_words))
