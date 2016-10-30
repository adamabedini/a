'''
Created on 31Oct.,2016

@author: Adam
'''
import sys
from math import log, sqrt
from itertools import combinations

import urllib
import nltk
import re
import numpy
import sys
import json

docslist = []   


def cosine_distance(a, b):
    cos = 0.0
    a_tfidf = a["tfidf"]
    for token, tfidf in b["tfidf"].items():
        if token in a_tfidf:
            cos += tfidf * a_tfidf[token]
    return cos

def normalize(features):
    norm = 1.0 / sqrt(sum(i**2 for i in features.values()))
    for k, v in features.items():
        features[k] = v * norm
    return features

def add_tfidf_to(documents):
    tokens = {}
    for id, doc in enumerate(documents):
        tf = {}
        doc["tfidf"] = {}
        doc_tokens = doc.get("tokens", [])
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        num_tokens = len(doc_tokens)
        if num_tokens > 0:
            for token, freq in tf.items():
                tokens.setdefault(token, []).append((id, float(freq) / num_tokens))

    doc_count = float(len(documents))
    for token, docs in tokens.items():
        idf = log(doc_count / len(docs))
        for id, tf in docs:
            tfidf = tf * idf
            if tfidf > 0:
                documents[id]["tfidf"][token] = tfidf

    #for doc in documents:
     #   doc["tfidf"] = normalize(doc["tfidf"])

def choose_cluster(node, cluster_lookup, edges):
    new = cluster_lookup[node]
    if node in edges:
        seen, num_seen = {}, {}
        for target, weight in edges.get(node, []):
            seen[cluster_lookup[target]] = seen.get(
                cluster_lookup[target], 0.0) + weight
        for k, v in seen.items():
            num_seen.setdefault(v, []).append(k)
        new = num_seen[max(num_seen)][0]
    return new

def majorclust(graph):
    cluster_lookup = dict((node, i) for i, node in enumerate(graph.nodes))

    count = 0
    movements = set()
    finished = False
    while not finished:
        finished = True
        for node in graph.nodes:
            new = choose_cluster(node, cluster_lookup, graph.edges)
            move = (node, cluster_lookup[node], new)
            if new != cluster_lookup[node] and move not in movements:
                movements.add(move)
                cluster_lookup[node] = new
                finished = False

    clusters = {}
    for k, v in cluster_lookup.items():
        clusters.setdefault(v, []).append(k)

    return clusters.values()

def get_distance_graph(documents):
    class Graph(object):
        def __init__(self):
            self.edges = {}

        def add_edge(self, n1, n2, w):
            self.edges.setdefault(n1, []).append((n2, w))
            self.edges.setdefault(n2, []).append((n1, w))

    graph = Graph()
    doc_ids = range(len(documents))
    graph.nodes = set(doc_ids)
    for a, b in combinations(doc_ids, 2):
        graph.add_edge(a, b, cosine_distance(documents[a], documents[b]))
    return graph


def remove_html_tags(text):
        """Remove html tags from a string"""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

def get_documents():

    
    list_of_lists = []
    listofwordsTmp = []
    listofwordsAll = []

    

    path=r'C:\inetpub\json.txt'
    
    with open(path,'r') as data_file:
        data = json.load(data_file)
    
        data = [doc for doc in data['hits']['hits']]
        for doc in data:
            #print("%s) %s" % (doc['_id'], doc['_source']['summaryid']))
            docslist.append(remove_html_tags(doc['_source']['searchabletext']))
            list_of_lists.append(remove_html_tags(doc['_source']['searchabletext']))
            
    return [{"text": doc, "tokens": doc.split()}
        for i, doc in enumerate(docslist)]

def main(args):
    documents = get_documents()
    add_tfidf_to(documents)
    dist_graph = get_distance_graph(documents)
    

    for cluster in majorclust(dist_graph):
        print ("===============================================================================================")
        print (cluster)
        for doc_id in cluster:
            print ([doc_id])
            print (docslist[doc_id])

if __name__ == '__main__':
    main(sys.argv)