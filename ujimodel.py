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
label_seq = ["anger","fear", "happy","love","sadness"]

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
data = pad_sequences(testSequences, maxlen=maxlen)

for k in range(num_folds):
    print('-' * 10)
    print("Fold %d/%d" % (k + 1, num_folds))
    validationSize = int(len(data) / num_folds)
    index1 = validationSize * k
    index2 = validationSize * (k + 1)

    xVal = data[index1:index2]
    # xVal2 = data2[index1:index2]
    yVal = labels[index1:index2]

    i = 10
    # get actual
    get_actual = yVal[i]  # get actual
    max_actual = np.amax(get_actual)
    index_actual = np.where(max_actual == max_actual)
    get_actual_label = label_seq[index_actual[0][0]]
    print("Actual Class :" + get_actual_label)

    # get predict
    get_predict = model.predictions[i]  # get predict
    max_predict = np.amax(get_predict)
    print(get_predict)
    index_predict = np.where(get_predict == max_predict)
    get_predict_label = label_seq[index_predict[0][0]]
    print("Predict Class :" + get_predict_label)


