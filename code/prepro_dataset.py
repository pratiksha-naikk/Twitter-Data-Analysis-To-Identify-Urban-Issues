import re,string
import itertools
import io
import numpy as np
import pandas as pd
import csv
from collections import Counter
from symspellpy.symspellpy import SymSpell, Verbosity  # import the module
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
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

#d = enchant.Dict('en_US')

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
    


makecols() #run this only when you want to make a csv file


'''
tweet = "I am Shaily I live in kharghar"
print(noProperNoun(tweet))



c=0
tweetsInList=[]
analyzedTweets=[]
originalTweets=[]
t=open('projecttweets.csv',encoding="utf8")
reader = csv.reader(t)
for row in reader:
    #using c so that it doesnt go through all 10k tweets rn
    if(c!=0):
        #c=0 is first row and it contains column names
        tweetsInList.append(row)
        analyzedTweets.append(row[2])
        originalTweets.append(row[2])
    c+=1
    if(c==10):
        #goes through only 1st 10 tweets
        break
t.close()

for entry in tweetsInList:
    #update cleaned text into list
    entry[2]=clean(entry[2])
    print("cleaned",entry[2])
c=0
for t in analyzedTweets:
    tweetsInList[c][2]=strip_all_entities(strip_links(t))
    c+=1


for entry in tweetsInList:
	check= entry[2]
	entry[2]=correct_spell(check)

for entry in tweetsInList:
    check= entry[2]
    entry[2]=preprocess(check)

for entry in tweetsInList:
    #insert sentiment into list
    entry.insert(5,getSentiment(entry[2]))
c=0
for row in tweetsInList:
    print("Original tweet: ",originalTweets[c])
    print("preprocessed tweet: ",row[2])
    print("complaint: ",row[3])
    print("Category: ",row[4])
    print("Sentiment TextBlob : ",row[5])
    c+=1

    #Not working enchant people dont like windows
def remove_repetitions(tweet):

    tweet=tweet.split()
    for i in range(len(tweet)):
        tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i])).replace('#', '')
        if len(tweet[i])>0:
            if not d.check(tweet[i]):
                tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet=' '.join(tweet)
    return tweet

def spellCorrect(tweet):
    #returns corrected tweet
    # max edit distance per lookup (per single word, not per whole input string)
    max_edit_distance_lookup = 2
    suggestions = sym_spell.lookup_compound(tweet,max_edit_distance_lookup)
    # display suggestion term, edit distance, and term frequency
    for suggestion in suggestions:
        return(suggestion.term)


def getSentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)



# maximum edit distance per dictionary precalculation
max_edit_distance_dictionary = 2
prefix_length = 7
    # create object
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    # load dictionary
sym_spell.load_dictionary('dict.txt', term_index=0, count_index=1)


'''