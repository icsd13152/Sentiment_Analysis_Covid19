import itertools

import dash as d
import joblib
import nltk
from dash.dependencies import Input, Output
import plotly.express as px
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


from sklearn.manifold import TSNE
import demoji

import pandas as pd


app = d.Dash(__name__)
data = pd.read_csv(r"Datasets/Corona_NLP_test.csv")

#
svm = joblib.load('savedModels/SVC.sav')
nb = joblib.load('savedModels/nb.sav')
lr = joblib.load('savedModels/LR.sav')
def getModel(model):
    if model == 'SVM':
        loaded_model = joblib.load('savedModels/SVC.sav')
        return loaded_model
    elif model == 'Naive Bayes':
        loaded_model = joblib.load('savedModels/nb.sav')
        return loaded_model
    elif model == 'Logistic Regression':
        loaded_model = joblib.load('savedModels/LR.sav')
        return loaded_model

tfidf_vectorizer = joblib.load('savedModels/vectorizer.sav')
# Regular expression for finding contractions
#short words/apostrophe lookup
contraction_dict1 = {"Â":"","’":"'","Ã":""}
contraction_dict2 = {"Â":"","’":"'","ain't": "are not","'s":" is","aren't": "are not","don't": "do not","Don't":"Do not",
                     "I'll":"I will","Didn't":"Did not","hasn't":"has not","NYC":"New York City","16MAR20":"",
                     "I'd":"I would","I've":"I have","you're":"you are","I'm":"I am","it's":"it is",
                     "#NZ":"","they'll":"they will","they're":"they are","can't":"can not","Y'all":"You All",
                     "I m":"I am","can't":"can not","don t":"do not","I ve":"I have","we're":"we are",
                     "LOL":"lough out loud","lol":"lough out loud","FYI":"For your information","OFC":"Of Course","ofc":"Of Course",
                     "#coronavirÃ¼s":"coronavirus","pls":"please","#stayhomesavelives":"stay home save lives",
                     "hasn't": 'has not',"haven't": 'have not',"he'd": 'he had / he would',"he'd've": 'he would have',
                     "he'll": 'he shall / he will',"he'll've": 'he shall have / he will have',
                     "he's": 'he has / he is',"how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will',
                     "how's": 'how has / how is', "i'd": 'I had / I would',"i'd've": 'I would have',  "i'll": 'I shall / I will',
                     "i'll've": 'I shall have / I will have',"i'm": 'I am', "i've": 'I have', "isn't": 'is not', "it'd": 'it had / it would',
                     "it'd've": 'it would have', "it'll": 'it shall / it will',
                     "it'll've": 'it shall have / it will have',"it's": 'it has / it is', "let's": 'let us',
                     "ma'am": 'madam', "mayn't": 'may not',
                     "might've": 'might have', "mightn't": 'might not',
                     "mightn't've": 'might not have', "must've": 'must have',"mustn't": 'must not',
                     "mustn't've": 'must not have', "needn't": 'need not',
                     "needn't've": 'need not have', "o'clock": 'of the clock',
                     "oughtn't": 'ought not', "oughtn't've": 'ought not have',
                     "shan't": 'shall not', "sha'n't": 'shall not',
                     "shan't've": 'shall not have', "she'd": 'she had / she would',
                     "she'd've": 'she would have', "she'll": 'she shall / she will',
                     "she'll've": 'she shall have / she will have',
                     "she's": 'she has / she is', "should've": 'should have',
                     "shouldn't": 'should not',"shouldn't've": 'should not have',
                     "so've": 'so have', "so's": 'so as / so is',
                     "that'd": 'that would / that had',"that'd've": 'that would have',
                     "that's": 'that has / that is', "there'd": 'there had / there would',
                     "there'd've": 'there would have', "there's": 'there has / there is',
                     "they'd": 'they had / they would',  "they'd've": 'they would have',
                     "they'll": 'they shall / they will', "they'll've": 'they shall have / they will have',
                     "they're": 'they are',  "they've": 'they have',
                     "to've": 'to have', "wasn't": 'was not',
                     "we'd": 'we had / we would',  "we'd've": 'we would have',
                     "we'll": 'we will', "we'll've": 'we will have',
                     "we're": 'we are', "we've": 'we have',
                     "weren't": 'were not', "what'll": 'what shall / what will',
                     "what'll've": 'what shall have / what will have',
                     "what're": 'what are', "what's": 'what has / what is',
                     "what've": 'what have',"when's": 'when has / when is',
                     "when've": 'when have', "where'd": 'where did',
                     "where's": 'where has / where is',
                     "where've": 'where have', "who'll": 'who shall / who will',
                     "who'll've": 'who shall have / who will have',
                     "who's": 'who has / who is', "who've": 'who have',
                     "why's": 'why has / why is', "why've": 'why have',
                     "will've": 'will have', "won't": 'will not',"won't've": 'will not have',
                     "would've": 'would have',"wouldn't": 'would not',"wouldn't've": 'would not have',
                     "y'all": 'you all', "y'all'd": 'you all would',
                     "y'all'd've": 'you all would have', "y'all're": 'you all are',
                     "y'all've": 'you all have', "you'd": 'you had / you would',
                     "you'd've": 'you would have',"&amp":"and","btc":"bitcoin","irs":"","spx":"","📍":"","✅":"","ive":"i have",
                     "coo":"","lka":"", "nyc":"","ktla":"","ppc":"pay per click","wjhl":"","plzzz":"please","orlf":"","etc":"",
                     "ktvu":"","amidst":"","biz":"business","djt":"","ict":"information communications technology","yep":"yes",
                     "yeap":"yes","gov":"goverment","psa":"public service announcement"
                     }

emoticons={':)': 'happy', ':‑)': 'happy',
           ':-]': 'happy', ':-3': 'happy',
           ':->': 'happy', '8-)': 'happy',
           ':-}': 'happy', ':o)': 'happy',
           ':c)': 'happy', ':^)': 'happy',
           '=]': 'happy', '=)': 'happy',
           '<3': 'happy', ':-(': 'sad',
           ':(': 'sad', ':c': 'sad',
           ':<': 'sad', ':[': 'sad',
           '>:[': 'sad', ':{': 'sad',
           '>:(': 'sad', ':-c': 'sad',
           ':-< ': 'sad', ':-[': 'sad',
           ':-||': 'sad','😢':'sad'}

myOwnStopWords={'price':"",
                'store':"",
                'supermarket':"",
                'food':"",
                'grocery':"",
                'people':"",
                'go':"",
                'consumer':""}


def lookup_dict(text, dictionary):
    for word in text.split():
        if word.lower() in dictionary:
            if word.lower() in text.split():
                text = text.replace(word, dictionary[word.lower()])
    return text

def get_pos( word ):
    w_synsets = wordnet.synsets(word)

    pos_counts = nltk.Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )

    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]

def process(DataProc):

    DataProc["Sentiment"] = DataProc["Sentiment"].replace('Extremely Negative', 'Negative', regex=True)
    #
    DataProc["Sentiment"] = DataProc["Sentiment"].replace('Extremely Positive', 'Positive', regex=True)
    # DataProc["Sentiment"]=DataProc["Sentiment"].replace('Negative', -1, regex=True)
    # DataProc["Sentiment"]=DataProc["Sentiment"].replace('Positive', 1, regex=True)
    # DataProc["Sentiment"]=DataProc["Sentiment"].replace('Neutral', 0, regex=True)
    #to lower case
    # DataProc["OriginalTweet"]=DataProc["OriginalTweet"].apply(lambda x:demoji.replace(x))
    #to lower case
    DataProc['OriginalTweet']  = DataProc['OriginalTweet'].str.lower()
    DataProc["OriginalTweet"] = DataProc["OriginalTweet"].apply(lambda x: lookup_dict(x,emoticons))

    DataProc['OriginalTweet']=DataProc['OriginalTweet'].apply(lambda x:lookup_dict(x,contraction_dict1))
    DataProc['OriginalTweet']=DataProc['OriginalTweet'].apply(lambda x:lookup_dict(x,contraction_dict2))
    DataProc['OriginalTweet']  = DataProc['OriginalTweet'].str.lower()
    DataProc['OriginalTweet']=DataProc['OriginalTweet'].apply(lambda x:lookup_dict(x,contraction_dict2))
    DataProc['OriginalTweet'] = DataProc['OriginalTweet'].apply(lambda x: ''.join(''.join(s)[:2] for _, s in itertools.groupby(x)))


    #to lower case
    DataProc['OriginalTweet']  = DataProc['OriginalTweet'].str.lower()

    #remove numbers
    DataProc["OriginalTweet"] = DataProc["OriginalTweet"].replace('[0-9]', '', regex=True)

    #remove mentions
    DataProc["OriginalTweet"] = DataProc["OriginalTweet"].replace('@([a-zA-Z0-9_]{1,50})', '', regex=True)

    #remove hashtags
    DataProc["OriginalTweet"] = DataProc["OriginalTweet"].replace('#', '', regex=True)

    #remove urls
    DataProc["OriginalTweet"] = DataProc["OriginalTweet"].replace('http\S+', '', regex=True)

    # # #remove all remaining bad chars
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace('[^\\w\\s]', '', regex=True)

    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("Â", "", regex=True)
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("â", "a", regex=True)
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("nz", "", regex=True)
    DataProc['OriginalTweet']  = DataProc['OriginalTweet'].str.strip()


    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("_", "", regex=True)
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("   ", " ", regex=True)
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("  ", " ", regex=True)
    # DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("rn", "", regex=True)
    # DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("vr", "", regex=True)
    # DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("hl", "", regex=True)
    # DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("u", "", regex=True)
    # DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("ot", "", regex=True)
    DataProc['OriginalTweet']  = DataProc['OriginalTweet'].str.strip()
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("coronavir¼", "coronavirus", regex=True)
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("pmmodi", "", regex=True)
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("amp", "and", regex=True)
    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].fillna(0)

    DataProc['OriginalTweet']  = DataProc['OriginalTweet'].str.strip()
    # print(DataProc["OriginalTweet"][16])
    # print(DataProc["OriginalTweet"].head(25))
    #Tokenize the tweets
    tokenized_tweets = DataProc["OriginalTweet"].apply(lambda x: x.split())

    #remove stopword(for example and,to at etc)
    stop_words = set(stopwords.words('english'))
    tokenized_tweets = tokenized_tweets.apply(lambda x: [word for word in x if not word in stop_words])

    # stemmer = PorterStemmer()#language='english'
    # tokenized_tweets= tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])

    lemmatizer = WordNetLemmatizer()
    tokenized_tweets = tokenized_tweets.apply(lambda x: [lemmatizer.lemmatize(i,get_pos( i )) for i in x])
    tokenized_tweets = tokenized_tweets.apply(lambda x: [demoji.replace(i) for i in x])
    tokenized_tweets = tokenized_tweets.apply(lambda x: [word for word in x if len(word)>2 or word=='go'])
    # tokenized_tweets= tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])

    #Joining the tokenized tweets
    for i in range(len(tokenized_tweets)):
        tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
    DataProc["OriginalTweet"] = tokenized_tweets


    DataProc['OriginalTweet']=DataProc['OriginalTweet'].apply(lambda x:lookup_dict(x,myOwnStopWords))
    all_words = []
    for line in list(DataProc['OriginalTweet']):
        words = line.split()
        for word in words:
            all_words.append(word.lower())


    DataProc["OriginalTweet"]=DataProc["OriginalTweet"].replace("  ", " ", regex=True)
    DataProc['OriginalTweet']  = DataProc['OriginalTweet'].str.strip()
    DataProc.drop_duplicates(subset ="OriginalTweet",
                             keep = 'last', inplace = True)
    DataProc.reset_index(drop=True, inplace=True)
    tfidf= tfidf_vectorizer.transform(DataProc["OriginalTweet"])
    return (tfidf,DataProc)

tweets,dataProcessed = process(data)

predictedLR = lr.predict(tweets)
predictedLRprobas = lr.predict_proba(tweets)

predictedSVM = svm.predict(tweets)
predictedSVMprobas = svm.predict_proba(tweets)

predictedNB = nb.predict(tweets)
predictedNBprobas = nb.predict_proba(tweets)

tsne = TSNE(n_components=1,learning_rate='auto',init='random')
X_embedded  = tsne.fit_transform(tweets)


dataProcessed['predictedSentimentSVM'] = predictedSVM

dataProcessed['predictedSentimentNB'] = predictedNB

dataProcessed['predictedSentimentLR'] = predictedLR

dataProcessed["predictedSentimentSVM"]=dataProcessed["predictedSentimentSVM"].replace(-1,'Negative', regex=True)
dataProcessed["predictedSentimentSVM"]=dataProcessed["predictedSentimentSVM"].replace( 1,'Positive', regex=True)
dataProcessed["predictedSentimentSVM"]=dataProcessed["predictedSentimentSVM"].replace( 0,'Neutral', regex=True)
dataProcessed["predictedSentimentLR"]=dataProcessed["predictedSentimentLR"].replace(-1,'Negative', regex=True)
dataProcessed["predictedSentimentLR"]=dataProcessed["predictedSentimentLR"].replace( 1,'Positive', regex=True)
dataProcessed["predictedSentimentLR"]=dataProcessed["predictedSentimentLR"].replace( 0,'Neutral', regex=True)
dataProcessed["predictedSentimentNB"]=dataProcessed["predictedSentimentNB"].replace(-1,'Negative', regex=True)
dataProcessed["predictedSentimentNB"]=dataProcessed["predictedSentimentNB"].replace( 1,'Positive', regex=True)
dataProcessed["predictedSentimentNB"]=dataProcessed["predictedSentimentNB"].replace( 0,'Neutral', regex=True)





models = ["Logistic Regression","SVM","Naive Bayes"]




app.layout = d.html.Div(className='row',children=[

    d.dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} for x in models],
        value= 'Logistic Regression',
        clearable=False,
    ),

    d.html.Div(
        d.html.Div(children=[
            d.dcc.Graph(id='left-top-bar-graph', style={'display': 'inline-block','width': '60vh', 'height': '50vh'}),
            d.dcc.Graph(id='right-top-pie-graph', style={'display': 'inline-block','width': '60vh', 'height': '50vh'})
        ])

    ),
    d.html.Div(
        d.html.Div(children=[
            d.dcc.Graph(id="bigrams-scatter",style={'width':'170vh'})

        ])

    ),


])

def calc(model,dataProcessed):
    countNegative = 0
    countPositive = 0
    countNeutral = 0
    if model == 'SVM':
        for i in range(len(dataProcessed['predictedSentimentSVM'])):
            # for j in range(len(processeddf["Sentiment"])):
            if dataProcessed['predictedSentimentSVM'][i] == dataProcessed["Sentiment"][i]:
                if dataProcessed['predictedSentimentSVM'][i] == 'Negative':
                    countNegative = countNegative + 1
                elif dataProcessed['predictedSentimentSVM'][i] == 'Positive':
                    countPositive = countPositive + 1
                elif dataProcessed['predictedSentimentSVM'][i] == 'Neutral':
                    countNeutral = countNeutral + 1
    elif model == 'Naive Bayes':
        for i in range(len(dataProcessed['predictedSentimentNB'])):
            # for j in range(len(processeddf["Sentiment"])):
            if dataProcessed['predictedSentimentNB'][i] == dataProcessed["Sentiment"][i]:
                if dataProcessed['predictedSentimentNB'][i] == 'Negative':
                    countNegative = countNegative + 1
                elif dataProcessed['predictedSentimentNB'][i] == 'Positive':
                    countPositive = countPositive + 1
                elif dataProcessed['predictedSentimentNB'][i] == 'Neutral':
                    countNeutral = countNeutral + 1
    elif model == 'Logistic Regression':
        for i in range(len(dataProcessed['predictedSentimentLR'])):
            # for j in range(len(processeddf["Sentiment"])):
            if dataProcessed['predictedSentimentLR'][i] == dataProcessed["Sentiment"][i]:
                if dataProcessed['predictedSentimentLR'][i] == 'Negative':
                    countNegative = countNegative + 1
                elif dataProcessed['predictedSentimentLR'][i] == 'Positive':
                    countPositive = countPositive + 1
                elif dataProcessed['predictedSentimentLR'][i] == 'Neutral':
                    countNeutral = countNeutral + 1
    return (countNegative,countNeutral,countPositive)

@app.callback(
    Output("left-top-bar-graph", "figure"),
    [Input("dropdown", "value")])
def update_bar_chart(model):

    countNegative,countNeutral,countPositive = calc(model,dataProcessed)
    fig = px.bar(x=['Negative','Actual Negative','Neutral','Actual Neutral','Positive','Actual Positive'],
                 y=[countNegative,len(dataProcessed[dataProcessed["Sentiment"]=='Negative']),countNeutral,len(dataProcessed[dataProcessed["Sentiment"]=='Neutral']),
                    countPositive,len(dataProcessed[dataProcessed["Sentiment"]=='Positive'])])


    return fig

@app.callback(
    Output("right-top-pie-graph", "figure"),
    [Input("dropdown", "value")])
def update_pieChart(model):
    countNegative,countNeutral,countPositive = calc(model,dataProcessed)
    fig2 = px.pie(values=[countNegative,countNeutral,countPositive], names=['Negative','Neutral','Positive'], title='Sentiments')

    return fig2






@app.callback(
    Output("bigrams-scatter", "figure"),
    [Input("dropdown", "value")],
)
def populate_bigram_scatter(model):
    predProbas = None
    df = pd.DataFrame(columns=['tsne_1','probas','Features'])
    if model == 'SVM':
        predProbas = predictedSVMprobas
        df['Sentiment'] = dataProcessed['predictedSentimentSVM']

    elif model == 'Naive Bayes':
        predProbas = predictedNBprobas
        df['Sentiment'] = dataProcessed['predictedSentimentNB']

    elif model == 'Logistic Regression':
        predProbas = predictedLRprobas
        df['Sentiment'] = dataProcessed['predictedSentimentLR']



    # countNegative,countNeutral,countPositive = calc(model,dataProcessed)

    df['tsne_1'] = X_embedded[:, 0]
    df['probas'] = pd.DataFrame(predProbas)
    df['tweet'] = dataProcessed['OriginalTweet']

    fig = px.scatter(df, x='probas', y='tsne_1', hover_name= 'tweet',color='Sentiment',size_max=45
                     , template='plotly_white', title='Bigram similarity per class', labels={'words': 'Avg. Length<BR>(words)'}
                     , color_continuous_scale=px.colors.sequential.Sunsetdark)
    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig
app.run_server(debug=False)