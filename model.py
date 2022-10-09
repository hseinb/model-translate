import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pickle
import logging

class FRENGTranslator:

    def __init__(self, model_path, en, fr):
        logging.info("FRENGTranslator class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")
        engfile = open('tokenizer_eng.pickle', 'rb')
        eng_tokenizer = Tokenizer()
        eng_tokenizer.open(engfile)
        frfile = open('tokenizer_fr.pickle', 'rb')
        fr_tokenizer = Tokenizer()
        fr_tokenizer.open(frfile)


    def predict(self, sentence):
        # load the image
        sentprep = self.prepare(sentence)

        # predict the class
        result = self.model.predict(sentprep)
        y_id_to_word = {value: key for key, value in fr_tokenizer.word_index.items()}
        y_id_to_word[0] = '<PAD>'
        final = ' '.join([y_id_to_word[np.argmax(x)] for x in result[0]])

        # return french statement
        return final

    def prepare(self, sent):
        # Prepare eng sentance to pass through model
        sentence = eng_tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=max_eng, padding='post')
        logging.info("Preparing the following statement {}".format(sent))
        return sentence


def main():
	model = FRENGTranslator('Bidir_model.h5')
	predicted = model.predict("she is driving the truck")
	logging.info("This translates to {}".format(predicted)) 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()