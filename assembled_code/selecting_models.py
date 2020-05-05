import io
import numpy as np
import pandas as pd
import csv
from collections import Counter
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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.naive_bayes import MultinomialNB
np.random.seed(500)


csvTweets=pd.read_csv("preprocessed_tweets_dataset.csv", encoding='utf-8')

# try this for every level of preprocessing- original tweet, cleaned,hashat,wstopword,lemmatize,wproper,cslp,csp,cs
feature_col=['cleaned'] #feature
predicted=['complaint'] #target try for - complaint and category
X=csvTweets[feature_col].values
print(X.shape)
y=csvTweets[predicted].values
print(y.shape)


split_size=0.20
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=split_size,random_state=42)
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(X.ravel().astype('U'))
Train_X_Tfidf = Tfidf_vect.transform(X_train.ravel().astype('U'))
Test_X_Tfidf = Tfidf_vect.transform(X_test.ravel().astype('U'))

def naive_bayes_model():
	#NAIVE BAYES
	# fit the training dataset on the NB classifier
	Naive = naive_bayes.MultinomialNB()
	Naive.fit(Train_X_Tfidf,y_train)
	# predict the labels on validation dataset
	predictions_NB = Naive.predict(Test_X_Tfidf)
	# Use accuracy_score function to get the accuracy
	print("Naive Bayes Accuracy Score with columns(",feature_col, predicted,") -> ",accuracy_score(predictions_NB, y_test)*100)


def svm_model():
	# Classifier - Algorithm - SVM
	# fit the training dataset on the classifier
	SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	SVM.fit(Train_X_Tfidf,y_train)
	# predict the labels on validation dataset
	predictions_SVM = SVM.predict(Test_X_Tfidf)
	# Use accuracy_score function to get the accuracy
	print("SVM Accuracy Score with columns(",feature_col, predicted,")-> ",accuracy_score(predictions_SVM, y_test)*100)

def lr():
	lr=LogisticRegression()
	lr.fit(Train_X_Tfidf,y_train)
	y_pred=lr.predict(Test_X_Tfidf)
	print("LogisticRegression Accuracy Score with columns(",feature_col, predicted,") -> ",accuracy_score(y_pred, y_test)*100)

#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

def Sgd():
	sgd=SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)
	sgd.fit(Train_X_Tfidf,y_train)
	y_pred=sgd.predict(Test_X_Tfidf)
	print("Stochastic Gradient Descent Accuracy Score with columns(",feature_col, predicted,") -> ",accuracy_score(y_pred, y_test)*100)

from sklearn.neighbors import KNeighborsClassifier
def kneigh():
	kn=KNeighborsClassifier(n_neighbors=17)
	kn.fit(Train_X_Tfidf,y_train)
	y_pred=kn.predict(Test_X_Tfidf)
	print("KNeighborsClassifier Accuracy Score with columns(",feature_col, predicted,") -> ",accuracy_score(y_pred, y_test)*100)

from sklearn.tree import DecisionTreeClassifier
def decision():
	dtree=DecisionTreeClassifier(max_depth=10, random_state=101,max_features=None,min_samples_leaf=12)
	dtree.fit(Train_X_Tfidf,y_train)
	y_pred=dtree.predict(Test_X_Tfidf)
	print("DecisionTreeClassifier Accuracy Score with columns(",feature_col, predicted,") -> ",accuracy_score(y_pred, y_test)*100)

from sklearn.ensemble import RandomForestClassifier
def forest():
	rf=RandomForestClassifier(max_depth=5,random_state=0)
	rf.fit(Train_X_Tfidf,y_train)
	y_pred=rf.predict(Test_X_Tfidf)
	print("RandomForestClassifier Accuracy Score with columns(",feature_col, predicted,") -> ",accuracy_score(y_pred, y_test)*100)

naive_bayes_model()
svm_model()
lr()
Sgd()
kneigh()
decision()
forest()
