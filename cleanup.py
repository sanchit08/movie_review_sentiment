from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from os import listdir


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

#load the document and add it to vocab
def add_to_vocab(filename,vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

#run through each doc in the dataset
def process_docs(directory,vocab):
	for filename in listdir(directory):
		path = directory + '/' + filename
		add_to_vocab(path,vocab)

#create an instance of the class Counter and create the vocabulary
vocab = Counter()
tokens = []
count = []
process_docs('/home/sanchit/Documents/Machine Learing/Datasets/txt_sentoken/neg',vocab)
process_docs('/home/sanchit/Documents/Machine Learing/Datasets/txt_sentoken/pos',vocab)
for i,c in vocab.items():
    if c>=2:
        tokens.append(i)
        count.append(str(c))

#save tokens to a file
str = '\n'
data = str.join(tokens)
file = open('vocab.txt', 'w')
file.write(data)
file.close()
