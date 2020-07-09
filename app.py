from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json
import pandas as pd
import numpy as np
import csv
import re
import string
import nltk
# from flaskext.mysql import MySQL
import mysql.connector

mydb = mysql.connector.connect(host="localhost", user="root", passwd="", database="klasifikasi")
"""mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'klasifikasi'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)"""

# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["RUNFILES_DIR"] = "C:/Users/Adinda Dwi/PycharmProjects/env/plaidml/"

# from keras.utils import to_categorical
# from keras_preprocessing.sequence import pad_sequences
# from keras_preprocessing.text import Tokenizer

# import keras
# from keras.models import load_model


# mendeklarasikan project Flask ke dalam variabel app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('beranda.html')


"""ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER'] = '/uploads'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION"""


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    """path = csv.reader(open('dataset_demo.csv'))
    prepro = []
    result = []
    isi = []
    i = 0
    for row in path:
        t = str(row).strip('[]').strip("'")
        b = t.rsplit(";")
        isi.append((i, b[0], b[1]))
        i += 1"""
    # conn = mydb.connect()
    cursor = mydb.cursor()
    sql = "SELECT * FROM web"
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    return render_template('dataset.html', dataset=result)


@app.route('/insertData', methods=['POST'])
def insertData():
    # conn = mydb.connect()
    cursor = mydb.cursor()
    tweet = request.form['tweet']
    # sql = "INSERT INTO web (isi_tw) VALUES (%s)"
    # t = tweet
    cursor.execute("INSERT INTO web (isi_tw) VALUES (%s)", (tweet, ))
    mydb.commit()
    return redirect(url_for('dataset'))


@app.route('/updateData', methods=['POST'])
def updateData():
    # conn = mydb.connect()
    cursor = mydb.cursor()
    id = request.form['id']
    tweet = request.form['uptweet']
    # sql = "UPDATE web SET isi_tw=%s WHERE id=%s"
    # t = (tweet, id)
    cursor.execute("UPDATE web SET isi_tw=%s WHERE id=%s", (tweet, id, ))
    mydb.commit()
    return redirect(url_for('dataset'))


@app.route('/deleteData/<string:id>', methods=['GET'])
def deleteData(id):
    # conn = mydb.connect()
    cursor = mydb.cursor()
    # t = (id)
    cursor.execute("DELETE FROM web WHERE id=%s", (id, ))
    mydb.commit()
    return redirect(url_for('dataset'))


@app.route('/preprocessing', methods=['GET', 'POST'])
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
        i += 1
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


@app.route('/uji')
def uji():
    return render_template('uji.html', )


# untuk menjalankan web service aplikasi flask
if __name__ == '__main__':
    app.run(debug=True)