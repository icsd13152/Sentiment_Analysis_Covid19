from fileinput import filename
import numpy as np
import pandas as pd
from flask import Flask, request, url_for
from nltk import FreqDist, PorterStemmer,SnowballStemmer
from pandas.io import pickle
from sklearn.feature_extraction.text import CountVectorizer
from werkzeug.utils import redirect
import joblib
app = Flask(__name__)

@app.route('/index')
def hello_world():
    return redirect(url_for('index.html'))
@app.route('/success/<name>')
def success(name):
    return 'result ' + name

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
    if request.method == 'POST':
        text = request.form['nm']
        model = request.form['slct']
        print(model)
        if model == '0':
            # load the model from disk
            loaded_model = joblib.load('final_NB.sav')
            text = processText(text)
            result = loaded_model.predict(text)
            print(result)
            return redirect(url_for('success',name = "NB"))


    # else:
    #     user = request.args.get('nm')
    #     return redirect(url_for('success',name = user))


def processText(inputText):



    data = pd.read_csv(r"Corona_NLP_test.csv", encoding='latin-1')
    #to lower case
    data['OriginalTweet']  = data['OriginalTweet'].str.lower()

    #remove numbers
    data["OriginalTweet"] = data["OriginalTweet"].replace('[0-9]', '', regex=True)

    #remove mentions
    data["OriginalTweet"] = data["OriginalTweet"].replace('@([a-zA-Z0-9_]{1,50})', '', regex=True)

    #remove hashtags
    data["OriginalTweet"] = data["OriginalTweet"].replace('#', '', regex=True)

    #remove urls
    data["OriginalTweet"] = data["OriginalTweet"].replace('http\S+', '', regex=True)

    # # #remove all remaining bad chars
    data["OriginalTweet"]=data["OriginalTweet"].replace('[^\\w\\s]', '', regex=True)

    data["OriginalTweet"]=data["OriginalTweet"].replace("Â", "'", regex=True)


    #Tokenize the tweets
    tokenized_tweets = data["OriginalTweet"].apply(lambda x: x.split())


    #Stemming the words to remove words with similar meaning
    stemmer = SnowballStemmer(language='english')
    tokenized_tweets= tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])



    # tokenized_tweets = tokenized_tweets.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])



    #Joining the tokenized tweets
    for i in range(len(tokenized_tweets)):
        tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
    data["OriginalTweet"] = tokenized_tweets


    # remove stopwords in order to have only the full-meaning words (remove and , to etc)
    # data["OriginalTweet"] = data["OriginalTweet"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    vectorizer = joblib.load('vectorizer.sav')

    countVec = vectorizer.transform(data["OriginalTweet"])


    # inputText= inputText.lower()
    # inputText= inputText.replace('[0-9]', '')
    # inputText= inputText.replace('@([a-zA-Z0-9_]{1,50})', '')
    # inputText= inputText.replace('#', '')
    # inputText= inputText.replace('http\S+', '')
    # inputText= inputText.replace('[^\\w\\s]', '')
    # inputText= inputText.replace("Â", "'")
    # #Tokenize the tweets
    # tokenized_tweets = inputText.split()
    # print(tokenized_tweets)
    # #Stemming the words to remove words with similar meaning
    # stemmer = SnowballStemmer(language='english')
    # list= []
    # for word in tokenized_tweets:
    #     list.append(stemmer.stem(word))


    print(tokenized_tweets)
    #Joining the tokenized tweets
    # for i in range(len(list)):
    #     list[i] = ''.join(list[i])
    # inputText = list
    # print(inputText)

    print(countVec.shape)
    return countVec



if __name__ == '__main__':
    app.run(debug = True)


