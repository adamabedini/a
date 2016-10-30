'''
Created on 30Oct.,2016

@author: Adam Abedini
'''


list_of_lists = []
listofwordsTmp = []
listofwordsAll = []
docslist = []
import urllib
import nltk
import re
import numpy
import sys
import json


from pprint import pprint
from macpath import split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

list_of_lists = []
listofwordsTmp = []
listofwordsAll = []
docslist = []

def remove_html_tags(text):
        """Remove html tags from a string"""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)


    



class tfidf:
  def __init__(self):
    self.weighted = False
    self.documents = []
    self.corpus_dict = {}

  def addDocument(self, doc_name, list_of_words):
    # building a dictionary
    doc_dict = {}
    for w in list_of_words:
      doc_dict[w] = doc_dict.get(w, 0.) + 1.0
      self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

    # normalizing the dictionary
    length = float(len(list_of_words))
    for k in doc_dict:
      doc_dict[k] = doc_dict[k] / length

    # add the normalized document to the corpus
    self.documents.append([doc_name, doc_dict])





def addingdocument (tfidf__ ,doc_name, list_of_words):
    tfidf__.addDocument(doc_name, list_of_words)
    return tfidf__

def getsimilarity (tfidf__ , list_of_words):
    return tfidf__.similarities(list_of_words)
   

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux



class Document(object):
    _id = ""
    summaryid = ""
    listofwords = []
    def __init__(self, _id, summaryid, listofwords):
        self._id = _id
        self.summaryid = summaryid
        self.listofwords = listofwords

def make_imDocument(_id_new, summaryid_new, listofwords_new):
    imDocument = Document(_id_new, summaryid_new, listofwords_new)
    return imDocument

        
        
        
        
    
    return 2



path=r'C:\inetpub\json.txt'

with open(path,'r') as data_file:
    data = json.load(data_file)

    data = [doc for doc in data['hits']['hits']]
    for doc in data:
        #print("%s) %s" % (doc['_id'], doc['_source']['summaryid']))
        listofwordsTmp = split(doc['_source']['searchabletext'])
        listofwordsAll += listofwordsTmp
        d01 = make_imDocument(doc['_id'] ,doc['_source']['summaryid'], remove_html_tags(doc['_source']['searchabletext']))
        list_of_lists.append(d01)
        
        docslist.append(remove_html_tags(doc['_source']['searchabletext']))
 


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docslist)

true_k = 3
j = 0
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=10000, n_init=1)
model.fit(X)

#Matrix = [[0 for x in range(true_k+1)] for y in range(11)] 
Matrix = [[0 for x in range(11)] for y in range(true_k+1)]


order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print ("Cluster %d:" % i),
    j = 0
    for ind in order_centroids[i, :10]:
        Matrix[i][j] = terms[ind]
        j += 1
      
        print (' %s' % terms[ind]),
    print



docscore = []
scoretemp = 0
i = 0
for i in range(30):
    k = 0
    for k in range(true_k):
        j = 0
        scoretemp = 0
        for j in range(10):
            #print( docslist[i].count(Matrix[k][j]))
            scoretemp  += docslist[i].count(Matrix[k][j])
            j += 1
        print("document id: %d cluster: %d scored: %d" % (i, k,  scoretemp))
        scoretemp= 0
        k +=1
i+=1




