import pandas as pd
data = pd.read_csv('preprocessed_tweets_dataset.csv', error_bad_lines=False);
data_text = data[['tweet']]
data_text['category'] = data.category
documents = data_text
print(len(documents))
#print(documents[:15])


#preprocessing

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

#lemmatizing and stemming
'''
The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. For instance:

am, are, is -be
car, cars, car's, cars' - car
The result of this mapping of text will be something like:
the boy's cars are different colors -
the boy car be differ color
'''
if __name__ == '__main__':
	def lemmatize_stemming(text):
	    return stem(WordNetLemmatizer().lemmatize(text, pos='v'))
	def preprocess(text):
	    result = []
	    for token in gensim.utils.simple_preprocess(text):
	        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
	            result.append(lemmatize_stemming(token))
	    return result


	processed_docs =documents['tweet'].fillna('').astype(str).map(preprocess)
	#Bagofwords

	dictionary = gensim.corpora.Dictionary(processed_docs)
	count = 0
	for k, v in dictionary.iteritems():
	    print(k, v)
	    count += 1
	    if count > 10:
	        break
	#Filter out tokens
	dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

	# create a dictionary reporting how many words and how many times those words appear.
	bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

	#Create tf-idf model object on ‘bow_corpus’
	from gensim import corpora, models
	tfidf = models.TfidfModel(bow_corpus)
	corpus_tfidf = tfidf[bow_corpus]
	from pprint import pprint
	for doc in corpus_tfidf:
	    #pprint(doc)
	    break

	#Running LDA using Bag of Words
	#Train our lda model using gensim.models.LdaMulticore and save it to ‘lda_model’
	lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
	
	for idx, topic in lda_model.print_topics(-1):
	    print('Topic: {} \nWords: {}'.format(idx, topic))

	#Running LDA using TF-IDF
	lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
	for idx, topic in lda_model_tfidf.print_topics(-1):
	    print('Topic: {} Word: {}'.format(idx, topic))


	print(processed_docs[10])
	for index, score in sorted(lda_model[bow_corpus[10]], key=lambda tup: -1*tup[1]):
	    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

	for index, score in sorted(lda_model_tfidf[bow_corpus[10]], key=lambda tup: -1*tup[1]):
	    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
