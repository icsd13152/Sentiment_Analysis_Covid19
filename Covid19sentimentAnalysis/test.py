# import dash
import dash as d
# import dash as html

import joblib
from dash.dependencies import Input, Output
import plotly.express as px
from nltk import SnowballStemmer
import pandas as pd
# df = px.data.tips()
# days = df.day.unique()
from plotly.graph_objects import Layout
from plotly.validator_cache import ValidatorCache
app = d.Dash(__name__)
data = pd.read_csv(r"Corona_NLP_test.csv", encoding='latin-1')
def getModel(model):
    if model == 'SVM':
        loaded_model = joblib.load('final_SVC_rbf_CV.sav')
        return loaded_model
    elif model == 'Naive Bayes':
        loaded_model = joblib.load('final_NB_CountVec.sav')
        return loaded_model
    elif model == 'Logistic Regression':
        loaded_model = joblib.load('final_LR_CV.sav')
        return loaded_model

def process(data):


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

    vectorizer = joblib.load('Countvectorizer.sav')

    countVec = vectorizer.transform(data["OriginalTweet"])

    return countVec


def prediction(mymodel,tweets):
    predicted = mymodel.predict(tweets)

    data['predictedSentiment'] = predicted

    data["predictedSentiment"]=data["predictedSentiment"].replace(-1,'Negative', regex=True)
    data["predictedSentiment"]=data["predictedSentiment"].replace( 1,'Positive', regex=True)
    data["predictedSentiment"]=data["predictedSentiment"].replace( 0,'Neutral', regex=True)
    return data['predictedSentiment']



models = ["Naive Bayes","SVM","Logistic Regression"]




app.layout = d.html.Div(className='row',children=[

    d.dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} for x in models],
        value= 'Naive Bayes',
        clearable=False,
    ),

    d.html.Div(
        d.html.Div(children=[
            d.dcc.Graph(id='left-top-bar-graph', style={'display': 'inline-block'}),
            d.dcc.Graph(id='right-top-pie-graph', style={'display': 'inline-block'})
        ])

    ),



])

tweets = process(data)

@app.callback(
    Output("left-top-bar-graph", "figure"),
    [Input("dropdown", "value")])
def update_bar_chart(model):

    print(model)
    mymodel = getModel(model)
    prediction(mymodel,tweets)

    countNegative = 0
    countPositive = 0
    countNeutral = 0
    for i in data['predictedSentiment']:
        if i == 'Negative': countNegative = countNegative + 1
        elif i == 'Positive': countPositive = countPositive +1
        else: countNeutral = countNeutral +1


    fig = px.bar(x=['Negative','Neutral','Positive'], y=[countNegative,countNeutral,countPositive])


    return fig

@app.callback(
    Output("right-top-pie-graph", "figure"),
    [Input("dropdown", "value")])
def update_pieChart(model):
    mymodel = getModel(model)
    prediction(mymodel,tweets)
    countNegative = 0
    countPositive = 0
    countNeutral = 0
    for i in data['predictedSentiment']:
        if i == 'Negative': countNegative = countNegative + 1
        elif i == 'Positive': countPositive = countPositive +1
        else: countNeutral = countNeutral +1
    fig2 = px.pie(values=[countNegative,countNeutral,countPositive], names=['Negative','Neutral','Positive'], title='Sentiments')
    fig2.show()
    return fig2

app.run_server(debug=True)
