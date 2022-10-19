import nltk
import numpy as np

nltk.download('punkt')

from nltk.stem.porter import PorterStemmer

import pyttsx3

stemmer = PorterStemmer()


def Tokenize(sentence):
    return nltk.word_tokenize(sentence)


def Stem(word):
    return stemmer.stem(word.lower())


def BagOfWords(tokenized_sentence, all_words):
    tokenized_sentence = [Stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


# region ChatBot Voice
def ConvertTextToSpeech(text: str):
    engine = pyttsx3.init()

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    engine.say(text)
    engine.runAndWait()
    engine.stop()

# endregion
