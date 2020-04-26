import pandas as pd 
import io
import re,string
import itertools
from io import StringIO 

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
def remove_link(tweet):
	tweet = re.sub(r'http.?://[^\s]+[\s]?','', tweet)
	tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
	return(tweet)

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
    tweet = correct_spell(tweet)
    return tweet.strip()


df = pd.read_json(r'streamed_tweets.json',lines=True) 
df.to_csv()
data = df[['id','text']]
data['cleaned'] = data['text'].astype(str).apply(clean)
data.to_csv('preprocessed_streamed_tweets_1.csv',sep=',', encoding='utf-8')
print("STREAMED TWEETS CLEANED AND SAVED TO : preprocessed_streamed_tweets_1.csv ")