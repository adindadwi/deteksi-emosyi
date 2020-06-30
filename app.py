from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json
import pandas as pd
import numpy as np
import csv
import re
import string
import nltk

# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["RUNFILES_DIR"] = "C:/Users/Adinda Dwi/PycharmProjects/env/plaidml/"

from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

import keras
from keras.models import load_model


# mendeklarasikan project Flask ke dalam variabel app
app = Flask(__name__)

max_size = 5000  # 5000 kata teratas
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

model = load_model('model_SS_BED.h5')
dataset = pd.read_csv('data_train_processed_demo.csv', sep=';', header=None)
dataset.columns = ['label', 'tweet']
labels = dataset["label"].map({"anger": 0, "fear": 1, "happy": 2, "love": 3, "sadness": 4})
label_seq = ["anger", "fear", "happy", "love", "sadness"]


@app.route('/')
def index():
    return render_template('beranda.html')


"""ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER'] = '/uploads'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION"""


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    path = csv.reader(open('dataset_demo.csv'))
    prepro = []
    result = []
    isi = []
    i = 0
    for row in path:
        t = str(row).strip('[]').strip("'")
        b = t.rsplit(";")
        isi.append((i, b[0], b[1]))
        i += 1
    return render_template('dataset.html', dataset=isi)


@app.route('/preprocessing')
def preprocessing():
    # dt = pd.read_csv('dataset_demo.csv', sep=';', header=None)
    path = csv.reader(open('dataset_demo.csv'))
    prepro = []
    result = []
    isi = []
    i = 0
    for row in path:
        t = str(row).strip('[]').strip("'")
        b = t.rsplit(";")
        isi.append((i, b[0], b[1]))
        i+=1
    with open('preprocessing.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    for row in json_object:
        result.append((row["id"], row["tweet"], row["token"]))
    # print(dt['tweet'])
    if len(result) != len(isi):
        for line in isi:
            print(line[0])
            # data = []
            # line = line.append(data)
            tweet = str(line[2])
            # mengubah text menjadi lowercase (casefolding)
            lowc = tweet.lower()
            # menghapus angka
            numb = re.sub(r"\d+", "", lowc)
            # Menghapus tanda baca
            tanda = numb.translate(string.punctuation)
            # stopword
            stopwords = [line.rstrip() for line in open('stopword_list.txt')]
            stop = [a for a in tanda if a not in stopwords]
            stp = ''.join([str(elem) for elem in stop])
            # import StemmerFactory class
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()  # create Stemmer
            stemmer = factory.create_stemmer()
            stm = stemmer.stem(stp)  # stemming process
            # nltk tokenize
            token = nltk.tokenize.word_tokenize(stm)
            if (line[0], line[2], token) not in prepro:
                prepro.append(
                    {"id": line[0], "tweet": line[2], "token": token})

        with open('preprocessing.json', 'w') as outfile:
            json.dump(prepro, outfile)  # dump json menjadi file
        with open('preprocessing.json', 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        result = []
        for row in json_object:
            result.append((row["id"], row["tweet"], row["token"]))
    return render_template('preprocessing.html', preprocessing=result)


@app.route('/model')
def model():
    return render_template('model.html')


def getMetrics(predictions, ground):
    label2emotion = {0: "anger", 1: "fear", 2: "happy", 3: "love", 4: "sadness"}

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


metrics = {"accuracy": [],
           "precision": [],
           "recall": [],
           "f1": []}


@app.route('/uji')
def uji():
    return render_template('uji.html', )


# untuk menjalankan web service aplikasi flask
if __name__ == '__main__':
    app.run(debug=True)