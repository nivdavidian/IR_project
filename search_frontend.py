from flask import Flask, request, jsonify
from inverted_index_gcp import InvertedIndex
import numpy as np
from time import time
from corpus_data import CorpusData
from collections import Counter
import pickle
from google.cloud import storage
import re
import csv

import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords


nltk.download('stopwords')

BUCKET_NAME = "ex3ir205557564"

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

body_index:InvertedIndex = InvertedIndex.read_index('.', 'body_index')
title_index:InvertedIndex = InvertedIndex.read_index('.', 'title_index')
anchor_index:InvertedIndex = InvertedIndex.read_index('.', 'anchor_text_index')
corpus_d:CorpusData = CorpusData.read_from_blob("corpus_data", "corpus_data.pkl")
corpus_d.read_dls()
corpus_d.read_pr()

def tokenize(text):

    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [token for token in tokens if not token in all_stopwords]
    tokens = [token for token in tokens if token in title_index.df.keys() or token in body_index.df.keys()]
    return tokens

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    t_start = time()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    title_res_doc = search_title_out_binary(query)
    query = tokenize(query)
    b,k = 0.25, 2
    # title_res_doc = cos_sim(query, title_index,"title_postings_gcp")
    body_res_doc = bm_25(query,body_index,"body_postings_gcp", k, b)
    
    bw, tw= 0.3, 0.7
    all_res_doc = {}
    for doc_id in body_res_doc.keys():
        all_res_doc[doc_id] = bw*body_res_doc[doc_id]    
    for doc_id, res in title_res_doc.items():
        if all_res_doc.get(doc_id) == None:
            all_res_doc[doc_id] = res*tw
        else:
            all_res_doc[doc_id] += res*tw
            
    
    # page_rank_res_doc = page_rank_out(all_res_doc.keys())
    
    # for doc_id, res in page_rank_res_doc.items():
    #     all_res_doc[doc_id] += prw*res
    
    res = get_top_N(all_res_doc)
    # res = get_top_N(page_rank_out(all_res_doc.keys()))
    print(time()-t_start)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    t_start = time()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    
    doc_dict = cos_sim(query, body_index, "body_postings_gcp")
    
    res = [(doc_id, wij) for doc_id, wij in doc_dict.items()]
    res = sorted(res, key=lambda x: x[1], reverse=True)[:100]
    res = [(doc_id, corpus_d.id_to_title[doc_id]) for doc_id,_ in res]
    print(time()-t_start)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://34.135.249.255:8080/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    t_start = time()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    title_doc = search_title_out_binary(query)
    
    res = [(doc_id, corpus_d.id_to_title[doc_id]) for doc_id, _ in sorted(title_doc, key=lambda x:x[1], reverse=True)]
    print((time()-t_start))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    t_start = time()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_unique = frozenset(tokenize(query))
    num_distinct = len(query_unique)
    pls = [anchor_index.read_term_pl(w, "anchor_text_postings_gcp") for w in query_unique]
    doc_ids = np.unique([doc_id for pl in pls for doc_id, _ in pl])
    d = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    num_docs = len(doc_ids)

    vec = np.zeros(shape=(num_docs, num_distinct))
    for i, pl in enumerate(pls):
        for doc_id,_ in pl:
            vec[d[doc_id]][i] = 1
    
    res = [(int(doc_ids[i]), corpus_d.id_to_title[doc_ids[i]]) for i, _ in sorted(enumerate(np.sum(vec, axis=1)) , key=lambda x: x[1], reverse=True) if doc_ids[i] in corpus_d.id_to_title.keys()]
    print((time()-t_start))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    pr = corpus_d.read_pr()
    for wiki_id in wiki_ids:
        if pr.get(f"{wiki_id}") is not None:
            res.append(pr[f"{wiki_id}"])
        else:
            res.append("0")
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("page_views/page_views.pkl")
    with blob.open("rb") as f:
        pv_file = pickle.load(f)
    for idd in wiki_ids:
        res.append(pv_file[idd])
    # END SOLUTION
    return jsonify(res)

def bm_25(query, index, folder_name,k,b):
    N = corpus_d.N
    k,b = k, b
    unique_query = np.unique(query)
    doc_dict = {}
    avg_dl = sum(corpus_d.dl.values())/N
    max_bm = 0
    for q in unique_query:
        pl = index.read_term_pl(q, folder_name)
        for (doc_id, fr) in pl:
            if doc_id not in corpus_d.dl.keys():
                continue
            
            B = 1-b+b*(corpus_d.dl[doc_id]/avg_dl)
            tf = fr
            bm = (((k+1)*tf)/(B*k+tf))*np.log((N+1)/index.df[q])
            if doc_dict.get(doc_id) is None:
                doc_dict[doc_id] = bm
            else:
                doc_dict[doc_id] += bm
            if bm>max_bm:
                max_bm = bm
    
    for key in doc_dict.keys():
        doc_dict[key] = doc_dict[key]/max_bm
    
    # for doc_id, wij in doc_dict.items():
    #     doc_dict[doc_id] = wij* (1/len(query))*(1/corpus_d.dl[doc_id])
    
    return doc_dict

def cos_sim(query, index, folder_name):
    N = corpus_d.N
    unique_query = np.unique(query)
    doc_dict = {}
    for q in unique_query:
        pl = index.read_term_pl(q, folder_name)
        for (doc_id, fr) in pl:
            if doc_id not in corpus_d.dl.keys():
                continue
            if doc_dict.get(doc_id) is None:
                doc_dict[doc_id] = fr * np.log(N/body_index.df[q])
            else:
                doc_dict[doc_id] += fr * np.log(N/body_index.df[q])
    
    for doc_id, wij in doc_dict.items():
        doc_dict[doc_id] = wij* (1/len(query))*(1/corpus_d.dl[doc_id])
    
    return doc_dict

def page_rank_out(wiki_ids):
    # BEGIN SOLUTION
    res ={}
    pr = corpus_d.read_pr()
    # max = 0
    for wiki_id in wiki_ids:
        if pr.get(f"{wiki_id}") is not None:
            score = float(pr[f"{wiki_id}"])
            res[wiki_id] = score
            # if score > max:
            #     max = score
    
    # for key in res.keys():
    #     res[key] = res[key]/max
            
            
    return res
            

def search_title_out_binary(query):
    
    query_unique = frozenset(tokenize(query))
    num_distinct = len(query_unique)
    pls = [title_index.read_term_pl(w, "title_postings_gcp") for w in query_unique]
    doc_ids = np.unique([doc_id for pl in pls for doc_id, _ in pl])
    d = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    num_docs = len(doc_ids)

    vec = np.zeros(shape=(num_docs, num_distinct))
    for i, pl in enumerate(pls):
        for doc_id,_ in pl:
            vec[d[doc_id]][i] = 1
    
    vec = np.sum(vec, axis=1)/len(query_unique)
    return {int(doc_ids[doc_id_index]):res for doc_id_index, res in enumerate(vec)}

def get_top_N(res_dict,N=100):
    res = sorted([(doc_id, res) for doc_id, res in res_dict.items()], key=lambda x:x[1], reverse=True)
    #  print([i for i in res if i[0] =="61073786"])
    N = min(N,len(res))
    res = res[:N]
    res = [(x[0], corpus_d.id_to_title[x[0]]) for x in res]
    return res

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
