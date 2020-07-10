from app import app
import mysql.connector

mydb = mysql.connector.connect(host="us-cdbr-east-02.cleardb.com", user="b494071167088b", passwd="f589afc7", database="heroku_35abaa3e25ea5b0")