import pandas as pd
import csv
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import stream_tweets
import prepro_streamed
import visualization
np.random.seed(500)

csvTweets=pd.read_csv("preprocessed_tweets_dataset.csv", encoding='utf-8')
SVM_category = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM_complaint = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

Tfidf_vect = TfidfVectorizer(max_features=5000)

def train_category():
	feature_col=['hashat'] 
	predicted=['category'] 
	X=csvTweets[feature_col].values
	y=csvTweets[predicted].values
	split_size=0.20
	X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=split_size,random_state=42)
	Encoder = LabelEncoder()
	y_train = Encoder.fit_transform(y_train)
	y_test = Encoder.fit_transform(y_test)
	Tfidf_vect.fit(X.ravel().astype('U'))
	Train_X_Tfidf = Tfidf_vect.transform(X_train.ravel().astype('U'))
	Test_X_Tfidf = Tfidf_vect.transform(X_test.ravel().astype('U'))
	SVM_category.fit(Train_X_Tfidf,y_train)
	predictions_SVM = SVM_category.predict(Test_X_Tfidf)
	print("SVM Accuracy Score with columns(",feature_col, predicted,")-> ",accuracy_score(predictions_SVM, y_test)*100)
	

def category(tweet):
	tweet_vector=Tfidf_vect.transform([tweet])
	category=int(SVM_category.predict(tweet_vector))
	if(category==0):
		category= None #or other
	return(category)

def train_complaint():
	feature_col=['cleaned'] #feature
	predicted=['complaint'] 
	X=csvTweets[feature_col].values
	y=csvTweets[predicted].values
	split_size=0.20
	X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=split_size,random_state=42)
	Encoder = LabelEncoder()
	y_train = Encoder.fit_transform(y_train)
	y_test = Encoder.fit_transform(y_test)
	Tfidf_vect.fit(X.ravel().astype('U'))
	Train_X_Tfidf = Tfidf_vect.transform(X_train.ravel().astype('U'))
	Test_X_Tfidf = Tfidf_vect.transform(X_test.ravel().astype('U'))
	SVM_complaint.fit(Train_X_Tfidf,y_train)
	predictions_SVM = SVM_complaint.predict(Test_X_Tfidf)
	print("SVM Accuracy Score with columns(",feature_col, predicted,")-> ",accuracy_score(predictions_SVM, y_test)*100)

def complaint(tweet):
	tweet_vector=Tfidf_vect.transform([tweet])
	return(int(SVM_complaint.predict(tweet_vector)))

def analyze_streamed_tweets(data,result_csv_filename):
	train_complaint() #training svm model for complaint
	data['complaint']=data['cleaned'].astype(str).apply(complaint)
	train_category() #training svm model for category
	data['category']=data['hashat'].astype(str).apply(category)
	#print(data.tail(50))
	data.to_csv( result_csv_filename,sep=',', encoding='utf-8') #Final result saved here
	print("Analyzed tweets saved to:", result_csv_filename)
	return(data)
	

def streaming_tweets(fetched_json_ip,hashtags_ip):
	fetched_json= fetched_json_ip
	hashtags = hashtags_ip
	stream_tweets.get_tweets(fetched_json_ip,hashtags_ip)



streaming_tweets_filename='streamed_tweets_week1.json'
result_csv_filename = 'results_week1.csv'


hashtags_for_streaming=['MumbaiPolice', 'potholes', 'trains', 'airoli', 'navi mumbai', 'local trains', 'central railway', 'mumbai rains','garbage','violence', 'hawkers', 'noise', 'scam','harassment','electricity']
max_tweets=20 #number of tweets to stream
stream_tweets.get_tweets(streaming_tweets_filename,hashtags_for_streaming,max_tweets)
data= prepro_streamed.pre_streamed(streaming_tweets_filename)
result_data=analyze_streamed_tweets(data,result_csv_filename)
visualization.get_plot(result_data)




