from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json
import pandas as pd
import numpy as np
import csv
import re
import string
from flaskext.mysql import MySQL
# import mysql.connector

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["RUNFILES_DIR"] = "C:/Users/Adinda Dwi/PycharmProjects/env/plaidml/"

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model


# mendeklarasikan project Flask ke dalam variabel app
app = Flask(__name__)


"""mydb = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'b494071167088b'
app.config['MYSQL_DATABASE_PASSWORD'] = 'f589afc7'
app.config['MYSQL_DATABASE_DB'] = 'heroku_35abaa3e25ea5b0'
app.config['MYSQL_DATABASE_HOST'] = 'us-cdbr-east-02.cleardb.com'
mydb.init_app(app)"""

# mydb = mysql.connector.connect(host="localhost", user="root", passwd="", database="klasifikasi")

mydb = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'klasifikasi'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mydb.init_app(app)


# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["RUNFILES_DIR"] = "C:/Users/Adinda Dwi/PycharmProjects/env/plaidml/"

# from keras.utils import to_categorical
# from keras_preprocessing.sequence import pad_sequences
# from keras_preprocessing.text import Tokenizer

# import keras
# from keras.models import load_model


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
    conn = mydb.connect()
    curs = conn.cursor()
    sql = "SELECT * FROM web"
    curs.execute(sql)
    result = curs.fetchall()
    conn.close()
    """cursor = mydb.cursor()
    sql = "SELECT * FROM web"
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()"""
    return render_template('dataset.html', dataset=result)


@app.route('/insertData', methods=['POST'])
def insertData():
    conn = mydb.connect()
    curs = conn.cursor()
    tweet = request.form['tweet']
    lowc = tweet.lower()
    # menghapus angka
    numb = re.sub(r"\d+", "", lowc)
    # Menghapus tanda baca
    tanda = numb.translate(string.punctuation)
    # stopword
    stopwords = [line.rstrip() for line in open('stopword_list.txt')]  # gabung dr list horizon jadi list verti
    stop = [a for a in tanda if a not in stopwords]
    stp = ''.join([str(elem) for elem in stop])  # penggabungan
    # import StemmerFactory class
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()  # create Stemmer
    stemmer = factory.create_stemmer()
    stm = stemmer.stem(stp)  # stemming process
    # tokenize
    token = stm.split()
    hasil = str(token)

    sql = "INSERT INTO web (isi_tw, hasil_pre) VALUES (%s, %s)"
    t = tweet, hasil
    curs.execute(sql, t)
    conn.commit()

    return redirect(url_for('dataset'))


@app.route('/updateData', methods=['POST'])
def updateData():
    conn = mydb.connect()
    curs = conn.cursor()
    id = request.form['id']
    tweet = request.form['uptweet']
    sql = "UPDATE web SET isi_tw=%s WHERE id=%s"
    t = (tweet, id)
    curs.execute(sql, t)
    conn.commit()
    """cursor.execute("UPDATE web SET isi_tw=%s WHERE id=%s", (tweet, id, ))
    mydb.commit()"""
    return redirect(url_for('dataset'))


@app.route('/deleteData/<string:id>', methods=['GET'])
def deleteData(id):
    conn = mydb.connect()
    curs = conn.cursor()
    sql = "DELETE FROM web WHERE id=%s"
    t = (id)
    curs.execute(sql, t)
    conn.commit()
    """cursor.execute("DELETE FROM web WHERE id=%s", (id, ))
    mydb.commit()"""
    return redirect(url_for('dataset'))


@app.route('/preprocessing')
def preprocessing():
    # dt = pd.read_csv('dataset_demo.csv', sep=';', header=None)
    # path = csv.reader(open('dataset_demo.csv'))
    conn = mydb.connect()
    curs = conn.cursor()
    sql = "SELECT * FROM web"
    curs.execute(sql)
    result = curs.fetchall()
    conn.close()
    return render_template('preprocessing.html', preprocessing=result)


@app.route('/model')
def model():
    label_seq = ["anger", "fear", "happy", "love", "sadness"]
    maxlen = 100
    max_size = 5000
    # result = []
    # label_prediksi = []

    conn = mydb.connect()
    curs = conn.cursor()
    sql = "SELECT * FROM web ORDER BY id DESC LIMIT 1"
    # sql = "SELECT * FROM web"
    curs.execute(sql)
    isi_data = curs.fetchall()
    input = isi_data[0][2]
    id_web = isi_data[0][0]
    # print(id_web)
    # conn.close()

    """for i in isi_data:
        conn = mydb.connect()
        curs = conn.cursor()
        input = i[2]
        # input = []"""
    # sql = "SELECT * FROM uji ORDER BY id_uji DESC LIMIT 1"
    # curs.execute(sql)
    # id_cek = isi_data[0][0]

    # if id_web != id_cek:
    # else:
    input_text = input
    # print(input_text)
    tokenizer = Tokenizer(num_words=max_size)
    tokenizer.fit_on_texts([input_text])
    inputSequence = tokenizer.texts_to_sequences([input_text])
    input_data = pad_sequences(inputSequence, maxlen)
    # print(input_data)

    # predict
    model = load_model("model_5.h5")
    get_predict = model.predict(input_data)
    print(get_predict)

    max_predict = np.amax(get_predict)  # nilai paling besar dari prediksi = prediksi label
    index_predict = np.where(get_predict == max_predict)
    get_label = label_seq[index_predict[1][0]]
    # print("Predict Class :" + get_label)

    sql = "INSERT INTO uji (label_pred, id_web) VALUES (%s, %s)"
    t = get_label, id_web
    curs.execute(sql, t)
    conn.commit()

    sql = "SELECT id, isi_tw, label_pred FROM web JOIN uji WHERE id = id_web "
    curs.execute(sql)
    hasil = curs.fetchall()
    conn.close()

    return render_template('model.html', model=hasil)


@app.route('/uji')
def uji():
    return render_template('uji.html', )


# untuk menjalankan web service aplikasi flask
if __name__ == '__main__':
    app.run(debug=True)