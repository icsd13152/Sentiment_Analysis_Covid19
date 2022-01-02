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
#from wordcloud import WordCloud,STOPWORDS
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier

nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# import these modules
from nltk.stem import WordNetLemmatizer
# from nltk.stem.snowball import SnowballStemmer
import joblib
import time
from textblob import TextBlob
data = pd.read_csv(r"Corona_NLP_train.csv", encoding='ansi')
# data2 = pd.read_csv(r"finalSentimentdata2.csv", encoding='utf-8')
# print(data["Sentiment"].head(5))
# data['OriginalTweet'] = data['OriginalTweet']+data2['text']
# data2["sentiment"] = data2["sentiment"].replace('sad', 'Negative', regex=True)
# data2["sentiment"] = data2["sentiment"].replace('joy', 'Positive', regex=True)
# data['Sentiment'] = data['Sentiment']+data2['sentiment']
data["Sentiment"] = data["Sentiment"].replace('Extremely Negative', 'Negative', regex=True)

data["Sentiment"] = data["Sentiment"].replace('Extremely Positive', 'Positive', regex=True)

#transform Sentiment to number
#negative=0
#postive=1
#neutral=2
data["Sentiment"]=data["Sentiment"].replace('Negative', -1, regex=True)
data["Sentiment"]=data["Sentiment"].replace('Positive', 1, regex=True)
data["Sentiment"]=data["Sentiment"].replace('Neutral', 0, regex=True)
# print(data["Sentiment"].to_numpy())


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
                     "you'd've": 'you would have',"&amp":"and","btc":"bitcoin","irs":"","spx":"","📍":"","✅":""
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
 ':-||': 'sad',
  '😢':'sad'         }

myOwnStopWords={'price':"",
               'store':"",
               'supermarket':"",
               'food':"",
               'grocery':"",
               'people':"",
               'go':"",
               'consumer':"",
                'usdjpy':"", 'gbpusd':"", 'usdcnh':"", 'xauusd':"", 'wti':"", 'spx':"",'iave':"","aiave":""}
# Regular expression for finding contractions
# contractions_re1=re.compile('(%s)' % '|'.join(contraction_dict1.keys()))
# def expand_contractions1(text,contractions_dict=contraction_dict1):
#     def replace(match):
#
#         return contractions_dict[match.group(0)]
#     return contractions_re1.sub(replace, text)
# # # Expanding Contractions in the reviews
# emoticons = {'\\:‑\\)':"smiley"}#,":‑(":"sad", ":)":"smiley",":(":"sad",":P":"funny"
#
#
#
# contractions_re2=re.compile('(%s)' % '|'.join(contraction_dict2.keys()))
# def expand_contractions2(text,contractions_dict=contraction_dict2):
#     def replace(match):
#
#         return contractions_dict[match.group(0)]
#     return contractions_re2.sub(replace, text)

def lookup_dict(text, dictionary):
    if isinstance(text, float) == False and text is not None:
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
# data["OriginalTweet"]=data["OriginalTweet"].replace("rn", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("vr", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("hl", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("u", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("ot", "", regex=True)
data['OriginalTweet']  = data['OriginalTweet'].str.strip()
data["OriginalTweet"]=data["OriginalTweet"].replace("coronavir¼", "coronavirus", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("pmmodi", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("amp", "and", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("btc", "bitcoin", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("hand", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].fillna(0)

data['OriginalTweet']  = data['OriginalTweet'].str.strip()
# print(data["OriginalTweet"][16])
# print(data["OriginalTweet"].head(25))
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
data["OriginalTweet"]=data["OriginalTweet"].replace(" +", " ", regex=True)
#
# class_3,class_2,class_1 = data["Sentiment"].value_counts()
#
# c3 = data[data["Sentiment"] == -1]
# c2 = data[data["Sentiment"] == 1]
# c1 = data[data["Sentiment"] == 0]
# df_3 = c3.sample(class_1)
# df_2 = c2.sample(class_1)
# undersampled_df = pd.concat([df_3,df_2,c1],axis=0)
data['OriginalTweet']  = data['OriginalTweet'].str.strip()
# for i in data["OriginalTweet"]: print(i)

Y = data["Sentiment"]



tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_features=22000,ngram_range=(1,2))
tfidf = tfidf_vectorizer.fit_transform(data["OriginalTweet"])
# joblib.dump(tfidf_vectorizer,'tfidf_vectorizer.sav')

print(tfidf.toarray().shape)

from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn import feature_selection

# sns.heatmap(tfidf.todense()[:,np.random.randint(0,tfidf.shape[1],100)]==0,vmin=0,vmax=1,cbar=False).set_title('Sparse Matrix')
# plt.show()
X_names = tfidf_vectorizer.get_feature_names_out()
p_value_limit = 0.95
#this feature selection is ranking features with respect to their usefulness and is not used to make statements about statistical dependence or independence of variables.
features = pd.DataFrame()
for cat in np.unique(data["Sentiment"]):
    chi2, p = feature_selection.chi2(tfidf,data["Sentiment"]==cat)#chi2(tfidf,data["Sentiment"]==cat)
    features = features.append(pd.DataFrame({"feature":X_names,"score":1-p,"Y":cat}))
    features = features.sort_values(["Y","score"],ascending=[True,False])

    features = features[features['score']>p_value_limit]

X_scores = features["score"].unique().tolist()
X_names = features["feature"].unique().tolist()
print(X_names)


for cat in np.unique(data["Sentiment"]):
    print("# {}:".format(cat))
    print(" . selected features:", len(features[features["Y"]==cat]))
    print(" . top features:",",".join(features[features["Y"]==cat]["feature"].values[:20]))
    # print(" . top features scores:",",".join(str(features[features["Y"]==cat]["score"].values[:10])))
    print(" ")
# defining parameter range

tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names,ngram_range=(1,2))
tfidf_new = tfidf_vectorizer.fit_transform(data["OriginalTweet"])
joblib.dump(tfidf_vectorizer,'vectorizer.sav')
print(tfidf_new.toarray().shape)
dic_vocab = tfidf_vectorizer.vocabulary_




# sns.heatmap(tfidf_new.todense()[:,np.random.randint(0,tfidf_new.shape[1],100)]==0,vmin=0,vmax=1,cbar=False).set_title('Sparse Matrix')
# plt.show()

X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(tfidf_new, Y, test_size=0.25,shuffle=True,random_state=0)
from xgboost import XGBClassifier
# list_alpha = np.arange(1/100000, 10, 0.1)
# param_grid = {'alpha': list_alpha
#
#
#                }
# grid = GridSearchCV(naive_bayes.ComplementNB(), param_grid, refit = True, verbose = 5)
# #ComplementNB()
#
# grid.fit(X_train2, y_train2)
# # print best parameter after tuning
# print(grid.best_params_)
# grid_predictions = grid.predict(X_test2)
#
# # print classification report
# print(metrics.classification_report(y_test2, grid_predictions))
# from tabulate import tabulate
# f1scores=list()
# accs = list()
# info = {'Models': ['SVM RBF', 'Naive Bayes', 'Logistic Regression'], 'macro-F1 scores': f1scores, 'avg Accuracies': accs}
# print(tabulate(info,headers='keys',tablefmt='fancy_grid',showindex=True))
# param_grid = {'C':[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0,10,100,1000],
#                'max_iter':[100,200,300,1000],
#                'solver':['newton-cg', 'sag', 'saga', 'lbfgs']
#               }
# grid = GridSearchCV(LogisticRegression(multi_class='multinomial'), param_grid, refit = True, verbose = 5)
#
#
# grid.fit(X_train2, y_train2)
# # print best parameter after tuning
# print(grid.best_params_)
# grid_predictions = grid.predict(X_test2)
#
# # print classification report
# print(metrics.classification_report(y_test2, grid_predictions))

# clf4 = XGBClassifier().fit(X_train2,y_train2)
# y_pred = clf4.predict(X_test2)
# # predicted_prob = clf1.predict_proba(X_test2)
# m_confusion_test = metrics.confusion_matrix(y_test2, clf4.predict(X_test2))
# print("XGBoost")
# print(pd.DataFrame(data = m_confusion_test , columns=['Predicted -1', 'Predicted 0','Predicted 1'],index=['Predicted -1', 'Predicted 0','Predicted 1']))
# print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
# print(metrics.classification_report(y_pred,y_test2))
#


#
clf2 = naive_bayes.ComplementNB(alpha=1.2).fit(X_train2,y_train2)
y_pred = clf2.predict(X_test2)
predicted_prob2 = clf2.predict_proba(X_test2)
m_confusion_test = metrics.confusion_matrix(y_test2, clf2.predict(X_test2))
print("NB")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted -1', 'Predicted 0','Predicted 1'],index=['Actual -1','Actual 0','Actual 1']))
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
print(metrics.classification_report(y_pred,y_test2))
#
clf1 = svm.SVC(kernel='rbf',C=1000,gamma=0.01,probability=True).fit(X_train2,y_train2)
y_pred = clf1.predict(X_test2)
predicted_prob1 = clf1.predict_proba(X_test2)
m_confusion_test = metrics.confusion_matrix(y_test2, clf1.predict(X_test2))
print("SVM RBF")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted -1', 'Predicted 0','Predicted 1'],index=['Predicted -1', 'Predicted 0','Predicted 1']))
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
print(metrics.classification_report(y_pred,y_test2))
#
#
# clf3 = svm.SVC(kernel='linear',probability=True).fit(X_train2,y_train2)
# y_pred = clf3.predict(X_test2)
# # predicted_prob = clf1.predict_proba(X_test2)
# m_confusion_test = metrics.confusion_matrix(y_test2, clf3.predict(X_test2))
# print("SVM Linear")
# print(pd.DataFrame(data = m_confusion_test , columns=['Predicted -1', 'Predicted 0','Predicted 1'],index=['Predicted -1', 'Predicted 0','Predicted 1']))
# print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
# print(metrics.classification_report(y_pred,y_test2))





# classes = np.unique(y_test2)
# y_test_array = pd.get_dummies(y_test2, drop_first=False).values
#
# ## Accuracy, Precision, Recall
# accuracy = metrics.accuracy_score(y_test2, y_pred)
# auc = metrics.roc_auc_score(y_test2, predicted_prob1,
#                             multi_class="ovr")
# print("Accuracy:",  round(accuracy,2))
# print("Auc:", round(auc,2))
# print("Detail:")
#
#
# ## Plot confusion matrix
# cm = metrics.confusion_matrix(y_test2, y_pred)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
#             cbar=False)
# ax.set(xlabel="Predicted", ylabel="Actual", xticklabels=classes,
#        yticklabels=classes, title="Confusion matrix")
# plt.yticks(rotation=0)
#
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ## Plot roc
# for i in range(len(classes)):
#     fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],
#                                              predicted_prob1[:,i])
#     ax[0].plot(fpr, tpr, lw=3,
#                label='{0} (area={1:0.2f})'.format(classes[i],
#                                                   metrics.auc(fpr, tpr))
#                )
# ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
# ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05],
#           xlabel='False Positive Rate',
#           ylabel="True Positive Rate (Recall)",
#           title="Receiver operating characteristic")
# ax[0].legend(loc="lower right")
# ax[0].grid(True)
#
# ## Plot precision-recall curve
# for i in range(len(classes)):
#     precision, recall, thresholds = metrics.precision_recall_curve(
#         y_test_array[:,i], predicted_prob1[:,i])
#     ax[1].plot(recall, precision, lw=3,
#                label='{0} (area={1:0.2f})'.format(classes[i],
#                                                   metrics.auc(recall, precision))
#                )
# ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall',
#           ylabel="Precision", title="Precision-Recall curve")
# ax[1].legend(loc="best")
# ax[1].grid(True)
# plt.show()




# # meta_clf = RandomForestClassifier(criterion='gini', max_depth= 20, max_features= 'auto',  n_estimators= 500)
# meta_clf2 = svm.SVC(kernel='rbf',C=1,probability=True)
# meta_clf3 = LogisticRegression()

# meta_clf3 = XGBClassifier(use_label_encoder=False)
# sclf = StackingClassifier(classifiers=[clf1, clf2],use_probas = True, use_features_in_secondary = True,
#                           meta_classifier=meta_clf3)
# #
# print('3-fold cross validation:\n')
# #
# for clf, label in zip([clf1, clf2, sclf],
#                       ['SVM',
#                        'Naive Bayes',
#                        'StackingClassifier']):
#
#     scores = model_selection.cross_val_score(clf, X_train2, y_train2,
#                                              cv=3, scoring='accuracy')
#     print("Accuracy: %0.3f (+/- %0.3f) [%s]"
#           % (scores.mean(), scores.std(), label))
# #
# sclf.fit(X_train2, y_train2)   # training of stacked model
# pred_stack = model_stack.predict(X_test2)       # predictions on test data using stacked model
# acc_stack = metrics.accuracy_score(y_test2, pred_stack)  # evaluating accuracy
# print('accuracy score of Stacked model:', acc_stack * 100)

# param_grid3 = {'C':[1000],
#                'kernel':['linear']
#                }
#
# grid3 = GridSearchCV(svm.SVC(), param_grid3, refit = True, verbose = 5)
# grid3.fit(X_train2, y_train2)
# # print best parameter after tuning
# print(grid3.best_params_)
# grid_predictions = grid3.predict(X_test2)
#
# # print classification report
# print(metrics.classification_report(y_test2, grid_predictions))
# m_confusion_test = metrics.confusion_matrix(y_test2, grid3.predict(X_test2))
#
# print(pd.DataFrame(data = m_confusion_test , columns=['Predicted 0', 'Predicted 1','Predicted -1'],index=['Actual 0','Actual 1','Actual -1']))




filenameSVCLinear = 'fnb.sav'
joblib.dump(clf2, filenameSVCLinear)
filenameSVCLinearCV = 'SVC.sav'
joblib.dump(clf1, filenameSVCLinearCV)



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
start = time.time()
y_pred= clf1.predict(tfidf)
duration = time.time() - start
print("SVM")
print("Accuracy:",metrics.accuracy_score(data["Sentiment"], y_pred))
print(duration)

start2 = time.time()
y_pred2= clf2.predict(tfidf)
duration2 = time.time() - start2
print("NB")
print("Accuracy:",metrics.accuracy_score(data["Sentiment"], y_pred2))
print(duration2)

# start3 = time.time()
# y_pred3= sclf.predict(tfidf)
# duration3 = time.time() - start3
# print("Meta clf")
# print(duration3)

# print("Accuracy:",metrics.accuracy_score(data["Sentiment"], y_pred))
# print("Accuracy:",metrics.accuracy_score(data["Sentiment"], y_pred2))
# print("Accuracy:",metrics.accuracy_score(data["Sentiment"], y_pred3))



import tweepy as tw
import pandas as pd
from datetime import date, timedelta, datetime
import demoji
def getTweets():
    sysdateminus7 = ( datetime.now()- timedelta(days=7)).date()
    today = date.today()


    consumer_key= '6bq97Awgm0hbhz3avHkTLwlNC'
    consumer_secret= '1GnlUw7rpbWq5k5QdMaivbyv2KrE3Gkoa0PPwtmryKI1QArmJ0'
    access_token= '1458363905852551171-mLu1MIulnPZEHreJw0yDGBqaj8Cowv'
    access_token_secret= 'ozu44KBdeF6dlUxugXhjHyBzphmBCt5XMadChNb7M68nN'


    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Define the search term and the date_since date as variables
    search_words = '#Covid19 OR #covid19 OR #Covid-19 OR COVID-19'
    date_since = sysdateminus7
    #twitter has changed his API and we can only get tweets from 1 week ago.
    tweets = tw.Cursor(api.search_tweets,
                       q=search_words,
                       lang="en",
                       since=date_since,
                       until = today).items(20)

    listTweets = list()

    for tweet in tweets:
        if isinstance(tweet.text, float) == False and tweet.text is not None:
            listTweets.append(tweet.text)


    tweet_text = pd.DataFrame(data=listTweets,
                              columns=['tweet'])


    return tweet_text





data["tweet"] = getTweets()
# for i in data["tweet"]:
#     print(i)
data['tweet'].dropna()

data['tweet']=data['tweet'].apply(lambda x:lookup_dict(x,contraction_dict1))
data['tweet']=data['tweet'].apply(lambda x:lookup_dict(x,contraction_dict2))
data["tweet"] = data["tweet"].replace('nan', '', regex=True)
# data['tweet'] = data['tweet'].apply(lambda x: ''.join(''.join(s)[:2] for _, s in itertools.groupby(x)))


#to lower case
data['tweet']  = data['tweet'].str.lower()

#remove numbers
data["tweet"] = data["tweet"].replace('[0-9]', '', regex=True)

#remove mentions
data["tweet"] = data["tweet"].replace('@([a-zA-Z0-9_]{1,50})', '', regex=True)

#remove hashtags
data["tweet"] = data["tweet"].replace('#', '', regex=True)

#remove urls
data["tweet"] = data["tweet"].replace('http\S+', '', regex=True)

# # #remove all remaining bad chars
data["tweet"]=data["tweet"].replace('[^\\w\\s]', '', regex=True)

data["tweet"]=data["tweet"].replace("Â", "", regex=True)
data["tweet"]=data["tweet"].replace("â", "a", regex=True)
data["tweet"]=data["tweet"].replace("nz", "", regex=True)
data['tweet']  = data['tweet'].str.strip()


data["tweet"]=data["tweet"].replace("_", "", regex=True)
data["tweet"]=data["tweet"].replace("   ", " ", regex=True)
data["tweet"]=data["tweet"].replace("  ", " ", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("rn", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("vr", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("hl", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("u", "", regex=True)
# data["OriginalTweet"]=data["OriginalTweet"].replace("ot", "", regex=True)
data['tweet']  = data['tweet'].str.strip()
data["tweet"]=data["tweet"].replace("coronavir¼", "coronavirus", regex=True)
data["tweet"]=data["tweet"].replace("pmmodi", "", regex=True)
data["tweet"]=data["tweet"].replace("amp", "and", regex=True)
data["tweet"]=data["tweet"].fillna(0)

data['tweet']  = data['tweet'].str.strip()
# print(data["OriginalTweet"][16])
# print(data["OriginalTweet"].head(25))
#Tokenize the tweets

tokenized_tweets = data["tweet"].apply(lambda x:  x.split() if isinstance(x,float) == False else '')

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
data["tweet"] = tokenized_tweets


data['tweet']=data['tweet'].apply(lambda x:lookup_dict(x,myOwnStopWords))
all_words = []
for line in list(data['tweet']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())


data["tweet"]=data["tweet"].replace("  ", " ", regex=True)


tfidf= tfidf_vectorizer.transform(data["tweet"])


print(data["tweet"].head(2))
start2 = time.time()
y_pred2= clf2.predict(tfidf)
print(y_pred2)
duration2 = time.time() - start2
print("SVC")
print(duration2)

