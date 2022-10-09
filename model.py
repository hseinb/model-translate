import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
#from keras.preprocessing.sequence import pad_sequences
import pickle
import logging
import numpy as np

class FRENGTranslator:

    def __init__(self, model_path):
        logging.info("FRENGTranslator class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")
        eng_tokenizer = Tokenizer()
        fr_tokenizer = Tokenizer()
        


    def predict(self, sentence, en, fr):
        # load the image
        
        with open(fr, 'rb') as handle:
            fr_tokenizer = pickle.load(handle)
        sentprep = self.prepare(sentence,en)

        # predict the class
        result = self.model.predict(sentprep)
        y_id_to_word = {value: key for key, value in fr_tokenizer.word_index.items()}
        y_id_to_word[0] = '<PAD>'
        final = ' '.join([y_id_to_word[np.argmax(x)] for x in result[0]])

        # return french statement
        return final

    def prepare(self, sent, en):
        # Prepare eng sentance to pass through model
        with open(en, 'rb') as handle:
            eng_tokenizer = pickle.load(handle)

        sentence = eng_tokenizer.texts_to_sequences([sent])
        sentence = tf.keras.preprocessing.sequence.pad_sequences(sentence, maxlen=15, padding='post')
        logging.info("Preparing the following statement {}".format(sent))
        return sentence


def main():
	model = FRENGTranslator('Bidir_model.h5')
	predicted = model.predict("she is driving the truck",'tokenizer_eng.pickle','tokenizer_fr.pickle')
	logging.info("This translates to {}".format(predicted)) 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()