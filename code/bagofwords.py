from string import punctuation
from os import listdir
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import csv
from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


 
# load doc into memory
def load_doc():
	tweetsInList=[]
	t=open('tweets.csv',encoding="utf8")
	reader = csv.reader(t)
	for row in reader:
		tweetsInList.append(row)
	t.close()
	return(str(tweetsInList))
	

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens
 
# load doc and add to vocab
def add_doc_to_vocab(vocab):
	# load doc
	doc = load_doc()
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)
 
def min_occ(vocab):
	# keep tokens with a min occurrence 2
	min_occurance = 3
	tokens = [k for k,c in vocab.items() if c >= min_occurance]
	print(len(tokens))
	# save tokens to a vocabulary file
	save_list(tokens, 'vocab.txt')


# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

#filter out tokens not in the vocabulary
def process_docs(vocab):
	tweetsInList=[]
	cleanTweets=[]
	#complaint=[]
	complaintCategory=[]
	c=0
	t=open('totalprecols.csv',encoding="utf8")
	reader = csv.reader(t)
	for row in reader:
		tweetsInList.append(row)
	t.close()
	z=0
	for row in tweetsInList:
		# 1st row is only column headings
		if(z==0):
			z+=1
			continue
		if(row[3]==''):
			continue
		tokens = clean_doc(row[13])
		# filter by vocab
		tokens = [w for w in tokens if w in vocab]

		cleanTweets.insert(c,' '.join(tokens))

		#complaint.insert(c,int(row[3]))
		if(row[5]==''):
			cint=0 #was showing error with blank values
		else:
			cint=int(float(row[5])) #converting category to int because model me problem hoti hai
		complaintCategory.insert(c,cint)
		c+=1

	
	return(cleanTweets,complaintCategory)
	


# load the vocabulary
vocab=open("vocab.txt", "r")
vocab=str(vocab.read())
vocab = vocab.split()
vocab = set(vocab)
docs,complaintCategory=process_docs(vocab)
docs=array(docs)

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(docs)

# encode training data set
Xtrain = tokenizer.texts_to_matrix(docs[:7000], mode='freq')
ytrain = array(complaintCategory[:7000])


# encode training data set
Xtest = tokenizer.texts_to_matrix(docs[7000:], mode='freq')
ytest = array(complaintCategory[7000:])

n_words = Xtest.shape[1]
# define network
model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))

'''

DONE 
this was to define vocabulary
# define vocab
vocab = Counter()
# add all docs to vocab
add_doc_to_vocab(vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
#print(vocab.most_common(50))

#to save our list of vocab with occurance greater than 3
min_occ(vocab)

'''




'''
EXTRA FUNCTIONS
# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)
#process_docs('txt_sentoken/pos', vocab)
#process_docs('txt_sentoken/neg', vocab)
'''
