from app import app
# import mysql.connector

# mydb = mysql.connector.connect(host="us-cdbr-east-02.cleardb.com", user="b494071167088b", passwd="f589afc7", database="heroku_35abaa3e25ea5b0")

from flaskext.mysql import MySQL


mydb = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'b494071167088b'
app.config['MYSQL_DATABASE_PASSWORD'] = 'f589afc7'
app.config['MYSQL_DATABASE_DB'] = 'heroku_35abaa3e25ea5b0'
app.config['MYSQL_DATABASE_HOST'] = 'us-cdbr-east-02.cleardb.com'
mydb.init_app(app)