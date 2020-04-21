import csv 
import os
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy
#suppress tensorflow warning and error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#to read the file containing reviews
def process(doc,vocab):
    documents = []
    d = load_doc(doc)
    q = clean_doc(d)
    tokens = [w for w in q if w in vocab]
    documents.append(tokens)
    return documents

#load documents
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
#clean the documents
def clean_doc(doc):
    tokens = word_tokenize(doc)
    words = []
    for word in tokens:
        if word.isalpha():
            words.append(word) 
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]  
    token = [word for word in words if len(word)>2]            
    return token

#load the word index and save in a list
with open('worddict.csv') as f:
    reader = csv.reader(f)
    mydict = dict(reader)
word = []
num = []
for w,c in mydict.items():
    word.append(w)
    num.append(c)

#load the text file containin review
v = open('vocab.txt', 'r')
vocab = v.read()
vocab = vocab.split()

test = process('/home/sanchit/Documents/Machine Learing/Codes/Movie Review Sentiment/review.txt',vocab)

te = []
tes = []
#remove words with capital letter
for i in range(len(test)):
    for j in range(len(test[i])):
        if not test[i][j][0].isupper():
            tes.append(test[i][j])
    te.append(tes)

#convert the words to corresponding numbers generated during train.py
encoded = te
for i in range(len(te)):
    for j in range(len(te[i])):
        if te[i][j] in word:
            index = word.index(te[i][j])
            encoded[i][j] = int(num[index])

#pad the test sequence
xtest = pad_sequences(encoded, maxlen = 1395, padding = 'post')

#load the saved model
model  =  load_model('movie review.h5')
#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#predict data
pre = model.predict_classes(xtest)
if pre == 1:
    str = 'GOOD'
else:
    str = 'BAD'
print('This is a ' + str + ' movie')
