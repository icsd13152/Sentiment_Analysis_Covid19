from itertools import cycle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import nltk
from sklearn.metrics import roc_curve, roc_auc_score,auc
import matplotlib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re, string
from nltk import FreqDist, PorterStemmer,SnowballStemmer
#from wordcloud import WordCloud,STOPWORDS
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')

# import these modules
from nltk.stem import WordNetLemmatizer
# from nltk.stem.snowball import SnowballStemmer
import joblib


data = pd.read_csv(r"Corona_NLP_train.csv", encoding='latin-1')
# print(data["Sentiment"].head(5))

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
data['OriginalTweet']  = data['OriginalTweet'].str.strip()


data["OriginalTweet"]=data["OriginalTweet"].replace("_", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("   ", " ", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("  ", " ", regex=True)

data['OriginalTweet']  = data['OriginalTweet'].str.strip()

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
stemmer = PorterStemmer()
tokenized_tweets= tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])

#get the lemma
lemmatizer = WordNetLemmatizer()
tokenized_tweets = tokenized_tweets.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])



#Joining the tokenized tweets
for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
data["OriginalTweet"] = tokenized_tweets


class_3,class_2,class_1 = data["Sentiment"].value_counts()

c3 = data[data["Sentiment"] == -1]
c2 = data[data["Sentiment"] == 1]
c1 = data[data["Sentiment"] == 0]
# df_3 = c3.sample(class_1)
# df_2 = c2.sample(class_1)
# undersampled_df = pd.concat([df_3,df_2,c1],axis=0)

df_1 = c1.sample(class_2, replace=True)
# df_2_over = c2.sample(class_2, replace=True)
oversampled_df = pd.concat([c3,df_1,c2], axis=0)

Y = data["Sentiment"]
vectorizer = CountVectorizer(stop_words= 'english')
countVec = vectorizer.fit_transform(data["OriginalTweet"])
joblib.dump(vectorizer,'Countvectorizer.sav')
# tfidf_vectorizer = TfidfVectorizer(stop_words= 'english')
# tfidf = tfidf_vectorizer.fit_transform(data["OriginalTweet"])
# joblib.dump(tfidf_vectorizer,'tfidf_vectorizer.sav')




#with 2 different vectorizers
X_train, X_test, y_train, y_test = model_selection.train_test_split(countVec, data["Sentiment"], test_size=0.33,random_state=42)

# X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(tfidf, Y, test_size=0.33,random_state=42)


#Parameter tuning
#from 1/100000 to 20 with step 0.11
list_alpha = np.arange(1/100000, 20, 0.11) #smoothing. (Laplace/Lidstone) smoothing parameter (0 for no smoothing)

score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0

for alpha in list_alpha:
    bayes = OneVsRestClassifier(naive_bayes.MultinomialNB(alpha=alpha))
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test),average='micro') #average=micro for multiClass
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test),average='micro')
    count = count + 1
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns =
['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print(models.head(20))


best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]

best_index = models[models['Test Precision']>0.65]['Test Accuracy'].idxmax()
print(list_alpha[best_index])
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
models.iloc[best_index, :]
y_pred = bayes.predict(X_test)
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
print("Naive Bayes with countVectorizer")
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1','Predicted 2'],
                   index = ['Actual 0', 'Actual 1','Actual 2']))
#
print(metrics.classification_report(y_pred,y_test))
y_pred = bayes.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
filenameNB1 = 'final_NB_CountVec.sav'
joblib.dump(bayes, filenameNB1)
# #===========================================================================================

# list_alpha = np.arange(1/100000, 20, 0.11) #smoothing. (Laplace/Lidstone) smoothing parameter (0 for no smoothing)

# score_train = np.zeros(len(list_alpha))
# score_test = np.zeros(len(list_alpha))
# recall_test = np.zeros(len(list_alpha))
# precision_test= np.zeros(len(list_alpha))
# count = 0
# for alpha in list_alpha:
#     bayes2 = OneVsRestClassifier(naive_bayes.MultinomialNB(alpha=alpha))
#     bayes2.fit(X_train2, y_train2)
#     score_train[count] = bayes2.score(X_train2, y_train2)
#     score_test[count]= bayes2.score(X_test2, y_test2)
#     recall_test[count] = metrics.recall_score(y_test2, bayes2.predict(X_test2),average='micro') #average=micro for multiClass
#     precision_test[count] = metrics.precision_score(y_test2, bayes2.predict(X_test2),average='micro')
#     count = count + 1
# matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
# models = pd.DataFrame(data = matrix, columns =
# ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
# print(models.head(20))




# best_index = models['Test Precision'].idxmax()
# models.iloc[best_index, :]
# best_index = models[models['Test Precision']>0.65]['Test Accuracy'].idxmax()
# print(list_alpha[best_index])
# bayes2 = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
# bayes2.fit(X_train2, y_train2)
# models.iloc[best_index, :]

# m_confusion_test = metrics.confusion_matrix(y_test2, bayes2.predict(X_test2))
# print("Naive Bayes with tf-idf")
# print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1', 'Predicted 2'],
#                    index = ['Actual 0', 'Actual 1','Actual 2']))
# y_pred = bayes2.predict(X_test2)
# print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
# filenameNB2 = 'final_NB_tf-idf.sav'
# joblib.dump(bayes2, filenameNB2)

# #===============================NB END ========================================


#================================ OneVSRest.SVC===============================


clf = OneVsRestClassifier(svm.SVC(C = 1000, kernel = 'linear')).fit(X_train,y_train)
y_pred = clf.predict(X_test)
m_confusion_test = metrics.confusion_matrix(y_test, clf.predict(X_test))
print("One Vs Rest SVM with CountVec linear")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted 0', 'Predicted 1','Predicted -1'],index=['Actual 0','Actual 1','Actual -1']))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_pred,y_test))


# #SVM with countVec C=1000
# svc1 = svm.SVC(C=1000, kernel='linear',probability=True)
# svc1.fit(X_train,y_train)
# y_pred = svc1.predict(X_test)
# m_confusion_test = metrics.confusion_matrix(y_test, svc1.predict(X_test))
# print("SVM with CountVec linear")
# print(pd.DataFrame(data = m_confusion_test , columns=['Predicted 0', 'Predicted 1','Predicted 2'],index=['Actual 0','Actual 1','Actual 2']))
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print(metrics.classification_report(y_pred,y_test))

# #========================================================
svc2 = OneVsRestClassifier(svm.SVC(C=1000, kernel='rbf',probability=True))
svc2.fit(X_train,y_train)
y_pred = svc2.predict(X_test)
m_confusion_test = metrics.confusion_matrix(y_test, svc2.predict(X_test))
print("SVM with cv rbf")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted 0', 'Predicted 1','Predicted 2'],index=['Actual 0','Actual 1','Actual 2']))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_pred,y_test))
# #============================================================


filenameSVC1 = 'final_SVC_linear_CV.sav'
filenameSVC2 = 'final_SVC_rbf_CV.sav'
# filenameSVC3 = 'final_SVC_linear_tfidf.sav'
# # filenameSVC4 = 'final_SVC_rbf_tfidf.sav'

joblib.dump(svc2, filenameSVC2)
joblib.dump(clf, filenameSVC1)
# # joblib.dump(svc2, filenameSVC2)
# # joblib.dump(svc3, filenameSVC3)
# # joblib.dump(svc4, filenameSVC4)
# #========================================== END of SVM============================================================
# #
lr=OneVsRestClassifier(LogisticRegression(random_state = 0))
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
m_confusion_test = metrics.confusion_matrix(y_test, lr.predict(X_test))
print("Logistic Regression with CV")
print(pd.DataFrame(data = m_confusion_test , columns=['Predicted 0', 'Predicted 1','Predicted 2'],index=['Actual 0','Actual 1','Actual 2']))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_pred,y_test))


filenameLR1 = 'final_LR_CV.sav'
# filenameLR2 = 'final_LR2_tfidf.sav'
joblib.dump(lr, filenameLR1)
# joblib.dump(lr2, filenameLR2)
#========================================== END of LR============================================================
data = pd.read_csv(r"Corona_NLP_test.csv", encoding='latin-1')
# print(data["Sentiment"].head(5))

data["Sentiment"] = data["Sentiment"].replace('Extremely Negative', 'Negative', regex=True)

data["Sentiment"] = data["Sentiment"].replace('Extremely Positive', 'Positive', regex=True)

#transform Sentiment to number
#negative=0
#postive=1
#neutral=2
data["Sentiment"]=data["Sentiment"].replace('Negative', -1, regex=True)
data["Sentiment"]=data["Sentiment"].replace('Positive', 1, regex=True)
data["Sentiment"]=data["Sentiment"].replace('Neutral', 0, regex=True)
print(data.groupby('Sentiment').size())
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
data['OriginalTweet']  = data['OriginalTweet'].str.strip()


data["OriginalTweet"]=data["OriginalTweet"].replace("_", "", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("   ", " ", regex=True)
data["OriginalTweet"]=data["OriginalTweet"].replace("  ", " ", regex=True)

data['OriginalTweet']  = data['OriginalTweet'].str.strip()

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
stemmer = PorterStemmer()
tokenized_tweets= tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x])

#get the lemma
lemmatizer = WordNetLemmatizer()
tokenized_tweets = tokenized_tweets.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])



#Joining the tokenized tweets
for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
data["OriginalTweet"] = tokenized_tweets


X1 = vectorizer.transform(data["OriginalTweet"])
# X2 = tfidf_vectorizer.transform(data["OriginalTweet"])

#======================CV=======================
y_pred = bayes.predict(X1)
print("Accuracy NB:",metrics.accuracy_score(data["Sentiment"].to_numpy(), y_pred))

y_pred = clf.predict(X1)
print("Accuracy SVM linear:",metrics.accuracy_score(data["Sentiment"].to_numpy(), y_pred))

y_pred = svc2.predict(X1)
print("Accuracy SVM rbf:",metrics.accuracy_score(data["Sentiment"].to_numpy(), y_pred))

y_pred = lr.predict(X1)
print("Accuracy LR:",metrics.accuracy_score(data["Sentiment"].to_numpy(), y_pred))
#===============================================

#====================TFIDF========================
# y_pred = bayes2.predict(X1)
# print("Accuracy NB:",metrics.accuracy_score(data["Sentiment"].to_numpy(), y_pred))
#
# y_pred = svc3.predict(X1)
# print("Accuracy SVM linear:",metrics.accuracy_score(data["Sentiment"].to_numpy(), y_pred))
#
# y_pred = svc4.predict(X1)
# print("Accuracy SVM rbf:",metrics.accuracy_score(data["Sentiment"].to_numpy(), y_pred))
#
# y_pred = lr2.predict(X1)
# print("Accuracy LR:",metrics.accuracy_score(data["Sentiment"].to_numpy(), y_pred))
#=================================================
