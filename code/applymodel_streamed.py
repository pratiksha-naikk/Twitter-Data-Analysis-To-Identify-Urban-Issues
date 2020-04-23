import io
import re,string
#import enchant
import itertools
import numpy as np
import pandas as pd
import csv
from collections import Counter
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
np.random.seed(500)

csvTweets=pd.read_csv("preprocessed_tweets_dataset.csv", encoding='utf-8')
#print(list(csvTweets.columns.values))


# try this for every level of preprocessing- original tweet, cleaned,hashat,wstopword,lemmatize,wproper,cslp,csp,cs
feature_col=['cleaned'] #feature
predicted_complaint=['complaint'] #target try for - complaint and category
predicted_svm=['category']
X=csvTweets[feature_col].values
y_complaint=csvTweets[predicted_complaint].values
y_svm=csvTweets[predicted_svm].values


split_size=0.20
X_train, X_test, y_train_svm, y_test_svm=train_test_split(X,y_svm,test_size=split_size,random_state=42)
X_train, X_test, y_train_complaint, y_test_complaint=train_test_split(X,y_complaint,test_size=split_size,random_state=42)

Encoder = LabelEncoder()
y_train_complaint = Encoder.fit_transform(y_train_complaint)
y_test_complaint = Encoder.fit_transform(y_test_complaint)
y_train_svm = Encoder.fit_transform(y_train_svm)
y_test_svm = Encoder.fit_transform(y_test_svm)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(X.ravel().astype('U'))
Train_X_Tfidf = Tfidf_vect.transform(X_train.ravel().astype('U'))
Test_X_Tfidf = Tfidf_vect.transform(X_test.ravel().astype('U'))

#SVM for category feild
def svm_category(tweet):
	
	SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	SVM.fit(Train_X_Tfidf,y_train_svm)
	# predict the labels on validation dataset
	predictions_SVM = SVM.predict(Test_X_Tfidf)
	# Use accuracy_score function to get the accuracy
	#print("SVM Accuracy Score with columns(",feature_col, predicted,")-> ",accuracy_score(predictions_SVM, y_test_svm)*100)

	tweet_vector=Tfidf_vect.transform([tweet])
	return(int(SVM.predict(tweet_vector)))
'''
#SGD for complaint feild
def sgd_complaint(tweet):
	sgd=SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)
	sgd.fit(Train_X_Tfidf,y_train_complaint)
	y_pred=sgd.predict(Test_X_Tfidf)
	#print("Stochastic Gradient Descent Accuracy Score with columns(",feature_col, predicted,") -> ",accuracy_score(y_pred, y_test_complaint)*100)

	tweet_vector=Tfidf_vect.transform([tweet])
	return(int(sgd.predict(tweet_vector)))
'''
def svm_complaint(tweet):
	SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	SVM.fit(Train_X_Tfidf,y_train_complaint)
	# predict the labels on validation dataset
	predictions_SVM = SVM.predict(Test_X_Tfidf)
	# Use accuracy_score function to get the accuracy
	#print("SVM Accuracy Score with columns(",feature_col, predicted,")-> ",accuracy_score(predictions_SVM, y_test_svm)*100)

	tweet_vector=Tfidf_vect.transform([tweet])
	return(int(SVM.predict(tweet_vector)))

def analyze_streamed_tweets():
	data = pd.read_csv('preprocessed_streamed_tweets_1.csv', encoding="ISO-8859-1")
	print("complaint feild")
	data['complaint']=data['text'].astype(str).apply(svm_complaint)
	print("category feild")
	data['category']=data['text'].astype(str).apply(svm_category)
	data.to_csv( 'result_streamed_tweets_1.csv',sep=',', encoding='utf-8') #Final result saved here
	print("Analyzed tweets saved to result_streamed_tweets_1.csv")

analyze_streamed_tweets()


