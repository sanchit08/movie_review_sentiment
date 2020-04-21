# Movie Review Sentiment
Program to analyse the sentiment of a movie(good or bad) from the movie review using natural langauge processing

## Natural Language Processing
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.

In this example we use the nltk module to clean up the dataset and the tokenizer api from the keras module
#### Tokenizer
Tokenization is a necessary first step in many natural language processing tasks, such as word counting, parsing, spell checking, corpus generation, and statistical analysis of text. Tokenizer is a compact pure-Python executable program and module for tokenizing text.
It allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count etc.We use the tokenizer api to convert our input data from a string of words to sequence of numbers which is then padded to a uniform length


## Dataset
The dataset is the Large Movie Review Dataset often referred to as the IMDB dataset. To download the dataset go [here.](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

The Large Movie Review Dataset contains 25,000 highly polar moving reviews (good or bad) for training and the same amount again for testing. The problem is to determine whether a given moving review has a positive or negative sentiment. In addition to this there are a large number of unlabelled reviews as well

## Instructions
- Run the **cleanup.py** python file. This iterates through all the training data and cleans it by removing all numbers, [stopwords](https://pythonprogramming.net/stop-words-nltk-tutorial/), punctuation and storing only the words necessary for determining sentiment onto a text file called **vocab.txt**.
- Run the **train.py** python file. This takes all our training data(25000 bipolar reviews) and converts it into setneces with only meaningful words based on **vocab.txt**(note that even though in this case the step may not be required if we have additional training data to be added later the vocabulary we generated will be important) and tokenizes and pads the test amnd train data using the tokenizer api.The generated key value pair is stored as a csv file namely **worddict.csv** which is used for prediction in the next step This is then trained using a Sequential model and making use of the [word embedding layer in Keras](https://keras.io/layers/embeddings/). You can find out more about word embeddings [here](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/). The model is saved along with model architecture and weights

### Prediction
In the text file **review.txt** store the review of the movie you want to predict the sentiment of. Then run the **predict.py** python file which uses the saved model from the previous step and uses it to predict the output

## Conclusion
Here i have used the review for the movie Parasite which gives particularly good remarks and after training the model and predicting the output was successfully shown

## References
- [Stack Overflow](stackoverflow.com)
- [Machine Learning Mastery](machinelearingmastery.com)
