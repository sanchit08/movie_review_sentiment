from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from os import listdir
import numpy
import csv

#doc load
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#cleanup the document
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

#clean docs and load all docs to a list
def process_docs(directory, vocab):
	documents = []
	for filename in listdir(directory):
		path = directory + '/' + filename
		doc = load_doc(path)
		q = clean_doc(doc)
		tokens = [w for w in q if w in vocab]
		documents.append(tokens)
	return documents


v = open('vocab.txt', 'r')
vocab = v.read()
vocab = vocab.split()
#create a list with all positive and negative train data
positive = process('/home/sanchit/Documents/Machine Learing/Datasets/txt_sentoken/pos', vocab)
negative = process('/home/sanchit/Documents/Machine Learing/Datasets/txt_sentoken/neg', vocab)
train_docs = negative + positive


#create the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)
words = tokenizer.word_index
#write the dictionary with key, value pair into a csv file
with open('worddict.csv','w') as f:
    writer = csv.writer(f)
    for w,c in words.items():
        writer.writerow([w,c])

#convert the documents to sequence of integers
encoded_docs = tokenizer.texts_to_sequences(train_docs)
e = []
for i in range(len(encoded_docs)):
    e.append(len(encoded_docs[i]))
max =  max(e)
print(max)


#pad the sequences to create the finalised training dataset
xtrain = pad_sequences(encoded_docs, maxlen = max, padding = 'post')
y = []
for i in range(25000):
    if i<12500:
        y.append(0)
    else:
        y.append(1)
ytrain = numpy.array(y)


#load the test documents
p = process_docs('/home/sanchit/Documents/Machine Learing/Datasets/txt_sentoken/po', vocab)
n = process_docs('/home/sanchit/Documents/Machine Learing/Datasets/txt_sentoken/ne', vocab)
test = n + p
encode_docs = tokenizer.texts_to_sequences(test)


#create test sets
xtest = pad_sequences(encode_docs, maxlen = max, padding = 'post')
y = []
for i in range(400):
    if i<200:
        y.append(0)
    else:
        y.append(1)
ytest = numpy.array(y)

vocab_size = len(words) + 1
print(vocab_size)


#create the model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length = max))
model.add(Conv1D(filters = 32, kernel_size = 8, activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
print(model.summary())

#compile our model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#train our model
model.fit(xtrain, ytrain, epochs = 10, batch_size = 25, verbose = 1)
#evaluate
loss,acc = model.evaluate(xtest,ytest,verbose = 1)
print('Accuracy is:', end = '')
print((acc*100))
print('Loss is:', end = '')
print((loss*100))

#save the model
model.save('movie review.h5')

