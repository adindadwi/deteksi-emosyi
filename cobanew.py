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
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras import backend as K

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix, \
    accuracy_score

import pandas as pd
import numpy as np
import io
import re
import string
import nltk
from numpy import array

dataTrain = pd.read_csv('data_train_processed_demo.csv', sep=';', header=None)
print(dataTrain.shape)
dataTrain.head()
dataTrain.columns = ['label', 'tweet']

# menghilangkan row yg memiliki nilai null atau string kosong
dataTrain = dataTrain.dropna()
dataTrain = dataTrain[dataTrain.label.apply(lambda x: x !=" ")]
dataTrain = dataTrain[dataTrain.tweet.apply(lambda x: x !=" ")]

# print(data["tweet"][168])
labels = dataTrain["label"].map({"anger": 0, "fear": 1, "happy": 2, "love": 3, "sadness": 4})
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
tokenizer.fit_on_texts(dataTrain['tweet'])
trainSequences = tokenizer.texts_to_sequences(dataTrain['tweet'])
testSequences = tokenizer.texts_to_sequences(dataTrain['tweet'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(trainSequences, maxlen)  # kata yang kurang dr 100 diberi padding 0
# data2 = pad_sequences(trainSequences, maxlen) # mengubah lists of integers->2D integer tensor of shape (samples, maxlen)
# print(data.shape)
labels = to_categorical(np.array(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
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
embedding_matrix = np.zeros((max_size, embedding_dim))
for word, i in word_index.items():
    if i > max_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# embedding layer SSWE
"""embeddings_index2 = {}
fo = open('embedding-sswe/sswe-h.txt', 'r', encoding="utf8")
for line in fo:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index2[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index2))
# create a weight matrix for words in training docs
embedding_matrix2 = np.zeros((max_size, embedding_dim2))
for word, i in word_index.items():
    if i > max_size - 1:
        break
    else:
        embedding_vector2 = embeddings_index2.get(word)
        if embedding_vector2 is not None:
            embedding_matrix2[i] = embedding_vector2"""


# def buildmodel(embedding_matrix, embedding_matrix2):
def buildmodel(embedding_matrix):
    """model1_input = Input(shape=(maxlen,), dtype='int32', name='glove')
    model2_input = Input(shape=(maxlen,), dtype='int32', name='sswe')"""
    # glove
    embedding_layer1 = Embedding(input_dim=max_size,
                                 output_dim=100,
                                 input_length=maxlen,
                                 weights=[embedding_matrix],
                                 trainable=False)  # (model1_input)

    """# sswe
    embedding_layer2 = Embedding(input_dim=max_size,
                                 output_dim=50,
                                 input_length=maxlen,
                                 weights=[embedding_matrix2],
                                 trainable=False)(model2_input)"""

    model = Sequential()
    model.add(embedding_layer1)
    # model_glove.add(Dropout(rate=dropout))
    # model = LSTM(units=lstm_dim, dropout=0.3)(embedding_layer1)
    model.add(LSTM(64, dropout=dropout))
    # model_glove.add(Dropout(rate=dropout))
    # model2 = LSTM(units=lstm_dim, dropout=0.3)(embedding_layer2)

    """x = Concatenate(axis=-1)([model1, model2])

    x = Dropout(rate=dropout)(x)
    output = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs=[model1_input, model2_input], outputs=output)"""
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

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    for c in range(0, num_classes):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])

        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])

        f1 = 2 * ((recall * precision) / (precision + recall))
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    """ print("Get classification metrics")
    classes = np.unique(predictions)
    print("\nClassification Report \n", classification_report(yVal, predictions))
    print("\nConfusion Matrix \n", confusion_matrix(yVal, predictions, labels=classes))
    print("\nAccuracy Score \n", accuracy_score(yVal, predictions))"""

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    return accuracy, precision, recall, f1


"""metrics = {
    "f1_e": (lambda y_test, y_pred:
             f1_score(y_test, y_pred, average='micro', labels=[labels['anger'], labels['fear'], labels['happy'], labels['love'], labels['sadness']])),
    "precision_e": (lambda y_test, y_pred:
                    precision_score(y_test, y_pred, average='micro',
                                    labels=[labels['anger'], labels['fear'], labels['happy'], labels['love'], labels['sadness']])),
    "recall_e": (lambda y_test, y_pred:
                 recall_score(y_test, y_pred, average='micro',
                              labels=[labels['anger'], labels['fear'], labels['happy'], labels['love'], labels['sadness']]))
}"""

# Perform k-fold cross validation
metrics = {"accuracy": [],
           "precision": [],
           "recall": [],
           "f1": []}

print("Starting k-fold cross validation...")
for k in range(num_folds):
    print('-' * 10)
    print("Fold %d/%d" % (k + 1, num_folds))
    validationSize = int(len(data) / num_folds)
    index1 = validationSize * k
    index2 = validationSize * (k + 1)

    xTrain = np.vstack((data[:index1], data[index2:]))
    # xTrain2 = np.vstack((data2[:index1], data2[index2:]))
    yTrain = np.vstack((labels[:index1], labels[index2:]))
    xVal = data[index1:index2]
    # xVal2 = data2[index1:index2]
    yVal = labels[index1:index2]
    print("Building model...")

    # create model
    model = buildmodel(embedding_matrix)
    model.summary()
    # [xTrain, xTrain2]
    # [xVal, xVal2]
    model.fit(xTrain,
              yTrain,
              validation_data=(xVal, yVal),
              batch_size=batchs_size,
              epochs=NUM_EPOCHS)

    predictions = model.predict(xVal)
    accuracy, precision, recall, f1 = getMetrics(predictions, yVal)
    metrics["accuracy"].append(accuracy)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)
    # print(classification_report(yVal, predictions))

    """
    Y = dataTrain[dataTrain.columns.drop(['tweet', 'index'])]
    label_names = Y.columns
    for (i, label) in enumerate(label_names):
        print(f'For message category {label}:\n')
        print(classification_report(predictions[:, i], yVal[label]))
        
    prediction_label = labels[np.argmax(predictions[1])]
    print('Actual label:' + yVal.iloc[i])
    print("Predicted label: " + prediction_label)"""

print("\n============= Metrics =================")
print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"]) / len(metrics["accuracy"])))
print("Average Cross-Validation Precision : %.4f" % (sum(metrics["precision"]) / len(metrics["precision"])))
print("Average Cross-Validation Recall : %.4f" % (sum(metrics["recall"]) / len(metrics["recall"])))
print("Average Cross-Validation F1 : %.4f" % (sum(metrics["f1"]) / len(metrics["f1"])))
print("\n======================================")

model = buildmodel(embedding_matrix)
# model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=batchs_size)
model.save('model_demo.h5')
# model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

print("Creating solution file...")
testData = pad_sequences(testSequences, maxlen=maxlen)
predictions = model.predict(testData)
predictions = predictions.argmax(axis=1)
print("")
"""t = 0

for text in data['tweet']:
    i = 0
    print("Prediksi untuk \"%s\": " % text)
    for label in labels:
        print("\t%s ==> %f" % (label, predictions[t][i]))
        i = i + 1
    t = t + 1"""
"""predictions[0]
print('Label Prediksi: %s ' % (labels[np.argmax(predictions[0])]))
print('Label Asli: %s ' % (yVal[0]))"""
# predictions = predictions.argmax(axis=1)

with io.open("solution.csv", "a") as fout:
    fout.write('\t'.join(["label", "tweet"]) + '\n')
    with io.open("data_train_processed.csv") as fin:
        fin.readline()
        for lineNum, line in enumerate(fin):
            fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
            fout.write(labels[predictions[lineNum]] + '\n')
print("Completed. Model parameters: ")
print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" % (learning_rate, lstm_dim, dropout, batchs_size))