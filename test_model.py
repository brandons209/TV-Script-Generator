from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys, os
import helper

#load dictionaries for converting text to ints and back
word_to_int = helper.load_dict('word_to_int.pkl')
int_to_word = helper.load_dict('int_to_word.pkl')
sequence_length = helper.load_dict('sequence_length.pkl')

#load in model user specified
if sys.argv[1] == "" or sys.argv[1] == "-h":
    print("Usage: test_model.py /path/to/model")
    sys.exit(0)
elif not os.path.isfile(sys.argv[1]):
    print("Error, file does not exist.")
    sys.exit(1)

model = load_model(sys.argv[1])

#same functions from simpson_script_gen:
#helper function that instead of just doing argmax for prediction, actually taking a sample of top possible words
#takes a tempature which defines how many predictions to consider. lower means the word picked will be closer to the highest predicted word.
def sample(prediction, temp=1.0):
    if temp <= 0:
        return np.argmax(prediction)
    prediction = prediction[0]
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temp
    expo_prediction = np.exp(prediction)
    prediction = expo_prediction / np.sum(expo_prediction)
    probabilities = np.random.multinomial(1, prediction, 1)
    return np.argmax(prediction)

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
        output_word = int_to_word[sample(prediction, temp=0.3)]
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
