import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import nltk
from sklearn.metrics import roc_curve, roc_auc_score,auc
import matplotlib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
import re, string
from nltk import FreqDist, PorterStemmer, SnowballStemmer, Counter

from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
# from nltk.stem.snowball import SnowballStemmer
import joblib
import time
from textblob import TextBlob
data = pd.read_csv(r"Corona_NLP_train.csv", encoding='ansi')


data["Sentiment"] = data["Sentiment"].replace('Extremely Negative', 'Negative', regex=True)

data["Sentiment"] = data["Sentiment"].replace('Extremely Positive', 'Positive', regex=True)

#transform Sentiment to number
#negative=0
#postive=1
#neutral=2
data["Sentiment"]=data["Sentiment"].replace('Negative', -1, regex=True)
data["Sentiment"]=data["Sentiment"].replace('Positive', 1, regex=True)
data["Sentiment"]=data["Sentiment"].replace('Neutral', 0, regex=True)



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
                     "you'd've": 'you would have',"&amp":"and"
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
           ':-||': 'sad'}

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
data["OriginalTweet"] = data["OriginalTweet"].apply(lambda x: lookup_dict(x,emoticons))

data['OriginalTweet']=data['OriginalTweet'].apply(lambda x:lookup_dict(x,contraction_dict1))
data['OriginalTweet']=data['OriginalTweet'].apply(lambda x:lookup_dict(x,contraction_dict2))

data['OriginalTweet'] = data['OriginalTweet'].apply(lambda x: ''.join(''.join(s)[:2] for _, s in itertools.groupby(x)))


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

data["OriginalTweet"]=data["OriginalTweet"].replace("Â", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("â", "a", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("nz", "", regex=True)
data['OriginalTweet']  = data['OriginalTweet'].str.strip()


data["OriginalTweet"]=data["OriginalTweet"].replace("_", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("   ", " ", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("  ", " ", regex=True)
data['OriginalTweet']  = data['OriginalTweet'].str.strip()
data["OriginalTweet"]=data["OriginalTweet"].replace("coronavir¼", "coronavirus", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("pmmodi", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("amp", "and", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].fillna(0)

data['OriginalTweet']  = data['OriginalTweet'].str.strip()

#Tokenize the tweets
tokenized_tweets = data["OriginalTweet"].apply(lambda x: x.split())

#remove stopword(for example and,to at etc)
stop_words = set(stopwords.words('english'))
tokenized_tweets = tokenized_tweets.apply(lambda x: [word for word in x if not word in stop_words])

#Stemming the words
# stemmer = PorterStemmer()#language='english'
# tokenized_tweets= tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])

def get_pos( word ):
    w_synsets = wordnet.synsets(word)

    pos_counts = nltk.Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )

    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]
#get the lemma
lemmatizer = WordNetLemmatizer()

tokenized_tweets = tokenized_tweets.apply(lambda x: [lemmatizer.lemmatize(i,get_pos( i )) for i in x])

tokenized_tweets = tokenized_tweets.apply(lambda x: [word for word in x if len(word)>2 or word=='go'])


#Joining the tokenized tweets
for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
data["OriginalTweet"] = tokenized_tweets


data['OriginalTweet']=data['OriginalTweet'].apply(lambda x:lookup_dict(x,myOwnStopWords))
all_words = []
for line in list(data['OriginalTweet']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())


print(Counter(all_words).most_common(10))
data["OriginalTweet"]=data["OriginalTweet"].replace("  ", " ", regex=True)
Y = data["Sentiment"]

tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_features=15000,ngram_range=(1,3))#min_df=0.0001
tfidf = tfidf_vectorizer.fit_transform(data["OriginalTweet"])
print(tfidf.toarray().shape)


import seaborn as sns
from sklearn import feature_selection

# sns.heatmap(tfidf.todense()[:,np.random.randint(0,tfidf.shape[1],100)]==0,vmin=0,vmax=1,cbar=False).set_title('Sparse Matrix')
# plt.show()
X_names = tfidf_vectorizer.get_feature_names_out()
p_value_limit = 0.95

features = pd.DataFrame()
for cat in np.unique(data["Sentiment"]):
    chi2, p = feature_selection.chi2(tfidf,data["Sentiment"]==cat)
    features = features.append(pd.DataFrame({"feature":X_names,"score":1-p,"Y":cat}))
    features = features.sort_values(["Y","score"],ascending=[True,False])
    features = features[features['score']>p_value_limit]
X_names = features["feature"].unique().tolist()
print(X_names)

for cat in np.unique(data["Sentiment"]):
    print("# {}:".format(cat))
    print(" . selected features:", len(features[features["Y"]==cat]))
    print(" . top features:",",".join(features[features["Y"]==cat]["feature"].values[:15]))
    print(" ")
# defining parameter range
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
tfidf_new = tfidf_vectorizer.fit_transform(data["OriginalTweet"])
# print(tfidf_new.toarray().shape)
dic_vocab = tfidf_vectorizer.vocabulary_
# print(dic_vocab)

# sns.heatmap(tfidf_new.todense()[:,np.random.randint(0,tfidf_new.shape[1],100)]==0,vmin=0,vmax=1,cbar=False).set_title('Sparse Matrix')
# plt.show()

X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(tfidf_new, Y, test_size=0.25,shuffle=True,random_state=42)



clf1 = svm.SVC(kernel='rbf',C=1.3,probability=True).fit(X_train2,y_train2)
y_pred = clf1.predict(X_test2)
# predicted_prob = clf1.predict_proba(X_test2)
m_confusion_test = metrics.confusion_matrix(y_test2, clf1.predict(X_test2))
print("SVM RBF")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted -1', 'Predicted 0','Predicted 1'],index=['Predicted -1', 'Predicted 0','Predicted 1']))
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
print(metrics.classification_report(y_pred,y_test2))

clf2 = naive_bayes.ComplementNB(alpha=0.40).fit(X_train2,y_train2)
y_pred = clf2.predict(X_test2)
predicted_prob = clf2.predict_proba(X_test2)
m_confusion_test = metrics.confusion_matrix(y_test2, clf2.predict(X_test2))
print("NB")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted -1', 'Predicted 0','Predicted 1'],index=['Actual -1','Actual 0','Actual 1']))
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
print(metrics.classification_report(y_pred,y_test2))

clf3 = svm.SVC(kernel='linear',probability=True).fit(X_train2,y_train2)
y_pred = clf3.predict(X_test2)
# predicted_prob = clf1.predict_proba(X_test2)
m_confusion_test = metrics.confusion_matrix(y_test2, clf3.predict(X_test2))
print("SVM Linear")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted -1', 'Predicted 0','Predicted 1'],index=['Predicted -1', 'Predicted 0','Predicted 1']))
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
print(metrics.classification_report(y_pred,y_test2))


clf4 = KNeighborsClassifier().fit(X_train2,y_train2)
y_pred = clf4.predict(X_test2)
# predicted_prob = clf1.predict_proba(X_test2)
m_confusion_test = metrics.confusion_matrix(y_test2, clf4.predict(X_test2))
print("KNN")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted -1', 'Predicted 0','Predicted 1'],index=['Predicted -1', 'Predicted 0','Predicted 1']))
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
print(metrics.classification_report(y_pred,y_test2))



data = pd.read_csv(r"Corona_NLP_test.csv", encoding='ansi')
data["Sentiment"] = data["Sentiment"].replace('Extremely Negative', 'Negative', regex=True)

data["Sentiment"] = data["Sentiment"].replace('Extremely Positive', 'Positive', regex=True)
data["Sentiment"]=data["Sentiment"].replace('Negative', -1, regex=True)
data["Sentiment"]=data["Sentiment"].replace('Positive', 1, regex=True)
data["Sentiment"]=data["Sentiment"].replace('Neutral', 0, regex=True)
#to lower case
data["OriginalTweet"] = data["OriginalTweet"].apply(lambda x: lookup_dict(x,emoticons))

data['OriginalTweet']=data['OriginalTweet'].apply(lambda x:lookup_dict(x,contraction_dict1))
data['OriginalTweet']=data['OriginalTweet'].apply(lambda x:lookup_dict(x,contraction_dict2))

data['OriginalTweet'] = data['OriginalTweet'].apply(lambda x: ''.join(''.join(s)[:2] for _, s in itertools.groupby(x)))


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

data["OriginalTweet"]=data["OriginalTweet"].replace("Â", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("â", "a", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("nz", "", regex=True)
data['OriginalTweet']  = data['OriginalTweet'].str.strip()


data["OriginalTweet"]=data["OriginalTweet"].replace("_", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("   ", " ", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("  ", " ", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("rn", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("vr", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("hl", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("u", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("ot", "", regex=True)
data['OriginalTweet']  = data['OriginalTweet'].str.strip()
data["OriginalTweet"]=data["OriginalTweet"].replace("coronavir¼", "coronavirus", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("pmmodi", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("amp", "and", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].fillna(0)

data['OriginalTweet']  = data['OriginalTweet'].str.strip()
# print(data["OriginalTweet"][16])
# print(data["OriginalTweet"].head(25))
#Tokenize the tweets
tokenized_tweets = data["OriginalTweet"].apply(lambda x: x.split())

#remove stopword(for example and,to at etc)
stop_words = set(stopwords.words('english'))
tokenized_tweets = tokenized_tweets.apply(lambda x: [word for word in x if not word in stop_words])

# stemmer = PorterStemmer()#language='english'
# tokenized_tweets= tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])

lemmatizer = WordNetLemmatizer()
tokenized_tweets = tokenized_tweets.apply(lambda x: [lemmatizer.lemmatize(i,get_pos( i )) for i in x])

tokenized_tweets = tokenized_tweets.apply(lambda x: [word for word in x if len(word)>2 or word=='go'])
# tokenized_tweets= tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])

#Joining the tokenized tweets
for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
data["OriginalTweet"] = tokenized_tweets


data['OriginalTweet']=data['OriginalTweet'].apply(lambda x:lookup_dict(x,myOwnStopWords))
all_words = []
for line in list(data['OriginalTweet']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())


data["OriginalTweet"]=data["OriginalTweet"].replace("  ", " ", regex=True)
# remove stopwords in order to have only the full-meaning words (remove and , to etc)
# data["OriginalTweet"] = data["OriginalTweet"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))



# countVec = vectorizer.transform(data["OriginalTweet"])
tfidf= tfidf_vectorizer.transform(data["OriginalTweet"])
# start = time.time()
# y_pred= clf1.predict(tfidf)
# duration = time.time() - start
# print("SVM")
# print(duration)

start2 = time.time()
y_pred2= clf2.predict(tfidf)
duration2 = time.time() - start2
print("NB")
print(duration2)

# start3 = time.time()
# y_pred3= sclf.predict(tfidf)
# duration3 = time.time() - start3
# print("Meta clf")
# print(duration3)

# print("Accuracy:",metrics.accuracy_score(data["Sentiment"], y_pred))
print("Accuracy:",metrics.accuracy_score(data["Sentiment"], y_pred2))
# print("Accuracy:",metrics.accuracy_score(data["Sentiment"], y_pred3))