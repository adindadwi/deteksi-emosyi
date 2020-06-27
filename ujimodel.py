import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["RUNFILES_DIR"] = "C:/Users/Adinda Dwi/PycharmProjects/env/plaidml/"


import keras
from keras import optimizers
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.engine.saving import load_model
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras import backend as K

import pandas as pd
import numpy as np
import io


dataTest = pd.read_csv('data_train_processed_demo.csv', sep=';', header=None)
dataTest.columns = ['label', 'tweet']

labels = dataTest["label"].map({"anger": 0, "fear": 1, "happy": 2, "love": 3, "sadness": 4})
label2emotion = {0: "anger", 1: "fear", 2: "happy", 3: "love", 4: "sadness"}

max_size = 5000  # 1000 kata teratas
maxlen = 100
embedding_dim = 100
embedding_dim2 = 50
learning_rate = 0.03
num_folds = 5
num_classes = 5
NUM_EPOCHS = 3
batchs_size = 5
lstm_dim = 64
hidden_layer_dim = 30
dropout = 0.3

tokenizer = Tokenizer(num_words=max_size)  # load data sebagai list of integer
tokenizer.fit_on_texts(dataTest['tweet'])
testSequences = tokenizer.texts_to_sequences(dataTest['tweet'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

model = load_model('model_SS_BED.h5')
testData = pad_sequences(testSequences, maxlen=maxlen)
labels = to_categorical(np.array(labels))
print(labels)

for line in testData:
    predictions = model.predict_classes(testData)
    print(predictions, labels)

# print(labels[np.argmax(predictions[0])])

