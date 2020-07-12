from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json
import pandas as pd
import numpy as np
import csv
import re
import string
import nltk
from flaskext.mysql import MySQL
# from app import app
# from db import mydb
# import mysql.connector

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
    tweet = request.form['tweet']
    conn = mydb.connect()
    curs = conn.cursor()
    sql = "INSERT INTO web (isi_tw) VALUES (%s)"
    t = tweet
    curs.execute(sql, t)
    conn.commit()
    """cursor = mydb.cursor()
    tweet = request.form['tweet']
    # sql = "INSERT INTO web (isi_tw) VALUES (%s)"
    # t = tweet
    cursor.execute("INSERT INTO web (isi_tw) VALUES (%s)", (tweet, ))
    mydb.commit()"""
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
    prepro = []
    final = []
    """isi = []
    i = 0
    for row in path:
        t = str(row).strip('[]').strip("'")
        b = t.rsplit(";")
        isi.append((i, b[0], b[1]))
        i += 1"""
    with open('preprocessing.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    for row in json_object:
        final.append((row["id"], row["tweet"], row["token"]))
    # print(dt['tweet'])
    if len(result) != len(final):
        for line in result:
            print(line[0])
            # data = []
            # line = line.append(data)
            tweet = str(line[1])
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
            if (line[0], line[1], token) not in prepro:
                prepro.append({"id": line[0], "tweet": line[1], "token": token})

        with open('preprocessing.json', 'w') as outfile:
            json.dump(prepro, outfile)  # dump json menjadi file
        with open('preprocessing.json', 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        final = []
        for row in json_object:
            final.append((row["id"], row["tweet"], row["token"]))
            sql = "SELECT * FROM preprocessing"
            curs.execute(sql)
            text = curs.fetchall()
            if not text:
                for row in final:
                    id_pre = row[0]
                    token = row[2]
                    for row in token:
                        sql = "INSERT INTO preprocessing (hasil_pre, id_pre) VALUES(%s, %s)"
                        t = (row, id_pre)
                        curs.execute(sql, t)
            else:
                sql = "TRUNCATE preprocessing"
                curs.execute(sql)
                for row in final:
                    id_pre = row[0]
                    token = row[2]
                    for row in token:
                        sql = "INSERT INTO preprocessing (hasil_pre, id_pre) VALUES(%s, %s)"
                        t = (row, id_pre)
                        curs.execute(sql, t)
    conn.commit()
    curs.close()
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