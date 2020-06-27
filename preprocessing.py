import pandas as pd
import numpy as np
import re
import string
import nltk
from numpy import array

data = pd.read_csv('dataset_demo.csv', sep=';', header=None)
print(data.shape)
data.head()
data.columns = ['label', 'tweet']

# menghilangkan row yg memiliki nilai null atau string kosong
data = data.dropna()
# data = data[data.label.apply(lambda x: x !=" ")]
data = data[data.tweet.apply(lambda x: x !=" ")]

# print(data["tweet"][168])
# labels = data["label"].map({"anger": 0, "fear": 1, "happy": 2, "love": 3, "sadness": 4})


def clean_text(tweet):
    # mengubah text menjadi lowercase (casefolding)
    tweet = str(tweet)
    tweet = tweet.lower()
    # menghapus angka
    tweet = re.sub(r"\d+", "", tweet)
    # Menghapus tanda baca
    # tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    tweet = tweet.translate(string.punctuation)
    # stopword
    stopwords = [line.rstrip() for line in open('stopword_list.txt')]
    stop = [a for a in tweet if a not in stopwords]
    tweet = ''.join([str(elem) for elem in stop])
    # import StemmerFactory class
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()  # create Stemmer
    stemmer = factory.create_stemmer()
    tweet = stemmer.stem(tweet)  # stemming process
    # nltk tokenize
    tweet = nltk.tokenize.word_tokenize(tweet)
    return tweet


# tweet dimasukkan jadi parameter clean-text
data['tweet'] = data['tweet'].map(lambda x: clean_text(x))

print(data['tweet'][1])

# simpan hasil
"""data_save = pd.DataFrame(data)
# data_label = pd.DataFrame(np.array(labels))
# result = pd.concat([data_save, data_label], axis=1)
data_save.to_csv('data_train_processed.csv', sep=';', index=False)"""