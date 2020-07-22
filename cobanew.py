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
from keras import backend as K

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix, \
    accuracy_score

import pandas as pd
import numpy as np
import joblib
import pickle
import io
import re
import string
from numpy import array

dataTrain = pd.read_csv('twitter_emotion_dataset3.csv', sep=';', header=None)
# print(dataTrain.shape)
dataTrain.head()
dataTrain.columns = ['label', 'tweet']
# print(data["tweet"][168])

labels = dataTrain["label"].map({"anger": 0, "fear": 1, "happy": 2, "love": 3, "sadness": 4})
label_seq = ["anger", "fear", "happy", "love", "sadness"]
label2emotion = {0: "anger", 1: "fear", 2: "happy", 3: "love", 4: "sadness"}

max_size = 5000  # 5000 kata teratas
maxlen = 100
embedding_dim = 100
# embedding_dim2 = 50
learning_rate = 0.03
num_folds = 5
num_classes = 5
NUM_EPOCHS = 3
batchs_size = 5
lstm_dim = 64
hidden_layer_dim = 30
dropout = 0.3


#tokenizer
tokenizer = Tokenizer(num_words=max_size)  # load data sebagai list of integer
# membuat index kamus berdasarkan frekuensi kata
tokenizer.fit_on_texts(dataTrain['tweet'])
# transform tiap teks menjadi sequence of integers
trainSequences = tokenizer.texts_to_sequences(dataTrain['tweet'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(trainSequences, maxlen)  # kata yang kurang dr 100 diberi padding 0
labels = to_categorical(np.array(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# print(labels)
# print(data)


# embedding layer GloVe
embeddings_index = {}
f = open('glove.6B/glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# create a weight matrix for words in training docs
# np.zeros = membuat matrix 0
embedding_matrix = np.zeros((max_size, embedding_dim))
for word, i in word_index.items():
    if i > max_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# def buildmodel(embedding_matrix, embedding_matrix2):
def buildmodel(embedding_matrix):
    # glove
    embedding_layer1 = Embedding(input_dim=max_size,
                                 output_dim=100,
                                 input_length=maxlen,
                                 weights=[embedding_matrix],
                                 trainable=False)  # (model1_input)

    model = Sequential()
    model.add(embedding_layer1)
    
    model.add(LSTM(64, dropout=dropout))

    model.add(Dense(units=num_classes, activation='softmax'))
    sgd = optimizers.sgd(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['acc'])
    return model


# metriks prediksi
def getMetrics(predictions, ground):
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    for c in range(0, num_classes):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])

        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])

        f1 = 2 * ((recall * precision) / (precision + recall))
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    return accuracy, precision, recall, f1


# Perform k-fold cross validation
metrics = {"accuracy": [],
           "precision": [],
           "recall": [],
           "f1": []}

print("Starting k-fold cross validation...")
for k in range(num_folds):
    print('-' * 10) # garis putus-putus
    print("Fold %d/%d" % (k + 1, num_folds))
    validationSize = int(len(data) / num_folds)
    index1 = validationSize * k
    index2 = validationSize * (k + 1)
    
    # biasa digunakan pada array 3D
    xTrain = np.vstack((data[:index1], data[index2:]))    
    yTrain = np.vstack((labels[:index1], labels[index2:]))
    xVal = data[index1:index2]
    yVal = labels[index1:index2]
    print("Building model...")

    # create model
    model = buildmodel(embedding_matrix)
    model.summary()

    print('Training model...')
    model.fit(xTrain,
              yTrain,
              validation_data=(xVal, yVal),
              batch_size=batchs_size,
              epochs=NUM_EPOCHS)
    print('Saving model...')
    model.save('model_%d.h5' % (num_folds))

    predictions = model.predict(xVal)

    accuracy, precision, recall, f1 = getMetrics(predictions, yVal)
    metrics["accuracy"].append(accuracy)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)
    # print(classification_report(yVal, predictions))
    

print("\n============= Metrics =================")
print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"]) / len(metrics["accuracy"])))
print("Average Cross-Validation Precision : %.4f" % (sum(metrics["precision"]) / len(metrics["precision"])))
print("Average Cross-Validation Recall : %.4f" % (sum(metrics["recall"]) / len(metrics["recall"])))
print("Average Cross-Validation F1 : %.4f" % (sum(metrics["f1"]) / len(metrics["f1"])))
print("\n======================================")

with open('try.pkl', 'wb') as model_file:
    joblib.dump(model, "classifier.pkl")
json_model = model.to_json()
with open('lstm_model.json', 'w') as json_file:
    json_file.write(json_model)


i = 10
# get actual
get_actual = yVal[i]  # get actual
max_actual = np.amax(get_actual)
index_actual = np.where(max_actual == max_actual)
get_actual_label = label_seq[index_actual[0][0]]
print("Actual Class :" + get_actual_label)

# get predict
get_predict = model.predict[i]  # get predict
max_predict = np.amax(get_predict)
print(get_predict)
index_predict = np.where(get_predict == max_predict)
get_predict_label = label_seq[index_predict[0][0]]
print("Predict Class :" + get_predict_label)