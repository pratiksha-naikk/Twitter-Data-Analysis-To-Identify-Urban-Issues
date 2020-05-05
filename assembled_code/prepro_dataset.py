import re,string
import itertools
import io
import numpy as np
import pandas as pd
import csv
from collections import Counter
from symspellpy.symspellpy import SymSpell, Verbosity 
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
#nltk.download('wordnet')
from stemming.porter2 import stem

dico = {}
dico1 = open('dicos/dico1.txt', 'rb')
for word in dico1:
    word = word.decode('utf8')
    word = word.split()
    dico[word[1]] = word[3]
dico1.close()
dico2 = open('dicos/dico2.txt', 'rb')
for word in dico2:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico2.close()
dico3 = open('dicos/dico2.txt', 'rb')
for word in dico3:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico3.close()

def correct_spell(tweet):

    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in dico.keys():
            tweet[i] = dico[tweet[i]]
    tweet = ' '.join(tweet)
    return tweet


def clean(tweet):

    #Separates the contractions and the punctuation
    tweet = remove_link(tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"\'ll","will",tweet)
    tweet = re.sub(r"\'re","are",tweet)
    tweet = re.sub(r"\'d","would",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"doesn't","does not",tweet)
    tweet = re.sub(r"[-()\";:<>{}+=?,]","",tweet)
    tweet = re.sub('\d+','', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+','',tweet)

    punct=string.punctuation
    transtab=str.maketrans(punct,len(punct)*' ')
    #tweet=tweet.translate(transtab)

    tweet = correct_spell(tweet)

    return tweet.strip()

def remove_link(tweet):
	tweet = re.sub(r'http.?://[^\s]+[\s]?','', tweet)
	tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
	return(tweet)

def remove_hashat(tweet):
    tweet = re.sub(r'@\w+','', tweet)
    tweet = re.sub(r'#\w+','', tweet)
    tweet = re.sub(r'http.?://[^\s]+[\s]?','', tweet)
    tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
    tweet = clean(tweet)
    return(tweet.strip())


def lemmatize_stemming(text):
    return stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    result=" ".join(str(x) for x in result)
    return str(result)

def stopwords(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)

    result=" ".join(str(x) for x in result)

    return str(result)

def noProperNoun(text):
    text=list(text.split(" "))
    #print(dico)
    c=0
    for i in text:
        if i not in dico.values():
            text.remove(i)
        c+=1
    text=" ".join(str(x) for x in text)

    return str(text)    
    
def makecols():
    data = pd.read_csv('tweets_dataset.csv', encoding="ISO-8859-1")
    
    print("CLEANING")
    data['cleaned']=data['tweet'].astype(str).apply(clean)
    print("REMOVING HASHTAG, @")
    data['hashat']=data['tweet'].astype(str).apply(remove_hashat)
    print("STOPWORDS")
    data['wstopword']=data['cleaned'].astype(str).apply(stopwords)
    print("LEMMATIZING")
    data['lemmatize']=data['wstopword'].astype(str).apply(preprocess)
    print("NO PROPER NOUN ")
    data['wproper']=data['cleaned'].astype(str).apply(noProperNoun)
    print("CLEAN, NO STOPWORDS, LEMMATIZING, NO PROPERNOUNS")
    data['cslp']=data['cleaned'].astype(str).apply(stopwords)
    data['cslp']=data['cslp'].astype(str).apply(preprocess)
    data['cslp']=data['cslp'].astype(str).apply(noProperNoun)
    print("CLEAN, NO STOPWORDS, NO PROPER NOUNS")
    data['csp']=data['cleaned'].astype(str).apply(stopwords)
    data['csp']=data['csp'].astype(str).apply(noProperNoun)
    print("CLEAN, NO STOPWORDS")
    data['cs']=data['cleaned'].astype(str).apply(stopwords)


    print(type(data))
    data.to_csv( 'preprocessed_tweets_dataset.csv',sep=',', encoding='utf-8') #Saved to this file
    


#makecols() #run this only when you want to make a csv file
