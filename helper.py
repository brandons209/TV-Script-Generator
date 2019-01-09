import numpy as np
import os, sys, glob
import pickle

"""
Loads text from specified path or directory, exits program if the file is not found.
"""
def load_script(path):

    if os.path.isfile(path):
        with open(path, 'r') as file:
            text = file.read()
        return text
    elif os.path.isdir(path):
        text = ""
        for file in glob.glob(path+"*.txt"):
            with open(file, 'r') as f:
                text = text + f.read()
        return text
    else:
        print("Error! {} is not a file or path!")
        sys.exit(1)


#saves dictionary to file in path for use with test_model.py
def save_dict(dict, path, filename):
    os.makedirs(path, exist_ok=True)
    dir = path + filename
    with open(dir, 'wb') as file:
        pickle.dump(dict, file)

#loads dictionary from file
def load_dict(path):
    with open(path, 'rb') as file:
         dict = pickle.load(file)
    return dict

#dictionaries for tokenizing puncuation and converting it back
punctuation_to_tokens = {'!':' ||exclaimation_mark|| ', ',':' ||comma|| ', '"':' ||quotation_mark|| ',
                          ';':' ||semicolon|| ', '.':' ||period|| ', '?':' ||question_mark|| ', '(':' ||left_parentheses|| ',
                          ')':' ||right_parentheses|| ', '--':' ||dash|| ', '\n':' ||return|| ', ':':' ||colon|| '}

tokens_to_punctuation = {token.strip(): punc for punc, token in punctuation_to_tokens.items()}

#for all of the puncuation in replace_list, convert it to tokens
def tokenize_punctuation(text):
    replace_list = ['.', ',', '!', '"', ';', '?', '(', ')', '--', '\n', ':']
    for char in replace_list:
        text = text.replace(char, punctuation_to_tokens[char])
    return text

#convert tokens back to puncuation
def untokenize_punctuation(text):
    replace_list = ['||period||', '||comma||', '||exclaimation_mark||', '||quotation_mark||',
                    '||semicolon||', '||question_mark||', '||left_parentheses||', '||right_parentheses||',
                    '||dash||', '||return||', '||colon||']
    for char in replace_list:
        if char == '||left_parentheses||':#added this since left parentheses had an extra space
            text = text.replace(' ' + char + ' ', tokens_to_punctuation[char])
        text = text.replace(' ' + char, tokens_to_punctuation[char])
    return text

"""
Takes text already converted to ints and a sequence length and returns the text split into seq_length sequences and generates targets for those sequences
"""
def gen_sequences(int_text, seq_length):
    seq_text = []
    targets = []
    for i in range(0, len(int_text) - seq_length, 1):
        seq_in = int_text[i:i + seq_length]
        seq_out = int_text[i + seq_length]
        seq_text.append([word for word in seq_in])
        targets.append(seq_out)#target is next word after the sequence
    return np.array(seq_text, dtype=np.int32), np.array(targets, dtype=np.int32)

"""
helper function that instead of just doing argmax for prediction, actually taking a sample of top possible words
takes a tempature which defines how many predictions to consider. lower means the word picked will be closer to the highest predicted word.
"""
def sample(prediction, temp=0):
    if temp <= 0:
        return np.argmax(prediction)
    prediction = prediction[0]
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temp
    expo_prediction = np.exp(prediction)
    prediction = expo_prediction / np.sum(expo_prediction)
    probabilities = np.random.multinomial(1, prediction, 1)
    return np.argmax(probabilities)
