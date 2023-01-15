from flask import Flask, request, jsonify
from inverted_index_gcp import InvertedIndex
import numpy as np
from time import time
from corpus_data import CorpusData
from collections import Counter
import pickle
from google.cloud import storage

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
corpus_d = CorpusData.read_from_blob("corpus_data", "corpus_data.pkl")
corpus_d.read_dls()

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
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [(1, "niv")]
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
    unique_query = np.unique(query)
    N = corpus_d.N
    pls = [body_index.read_term_pl(q, "body_postings_gcp") for q in unique_query]
    dl = corpus_d.dl
    df = body_index.df
    dis_doc_ids = np.unique([doc_id for pl in pls for (doc_id, _) in pl])
    doc_id_to_index = {doc_id:index for index, doc_id in enumerate(dis_doc_ids)}
    query_vec = np.zeros((len(unique_query)))
    doc_vec = np.zeros((len(dis_doc_ids), len(unique_query)))

    query_counter = Counter(query)
    for i,q in enumerate(unique_query):
        idf = np.log(N/len(pls[i]))
        tf = (query_counter[q]/len(query))
     #   print(N, len(pls[i]), q, tf, idf)
        query_vec[i] = tf*idf

    #print("doc length",dl[61073786])

    for i, pl in enumerate(pls):
        for doc_id, f in pl:
         #   if doc_id==61073786:
        #        print("frequency",f, unique_query[i])
            tf = f/dl[doc_id]
            idf = np.log(N/len(pl))
            doc_vec[doc_id_to_index[doc_id]][i] = tf*idf

    print(unique_query)
    print(query_vec)
    
    def calc_cosine_sim(Q, D):
        cos_sim = []

        tf_square = np.sum(np.square(D), axis=1) #shape(d,1)
        q_square = np.sum(np.square(Q)) #scalar
        mechane = np.sqrt(tf_square*q_square) #shape(d,1)
        mone = np.dot(D, Q) #(d,n) dot (n,1) = (d,1)
        for doc_id, cos in enumerate(mone/mechane):
            cos_sim.append((dis_doc_ids[doc_id],cos))

        return cos_sim


    def get_top_N(Q, D, N=100):
        res = sorted(calc_cosine_sim(Q,D), key=lambda x: x[1], reverse=True)
      #  print([i for i in res if i[0] =="61073786"])
        res = res[:N]
        res = [(f"{x[0]}", corpus_d.id_to_title[x[0]]) for x in res]
        return res
    

    res = get_top_N(query_vec, doc_vec)
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
    
    res = [(f"{doc_ids[i]}", corpus_d.id_to_title[doc_ids[i]]) for i, _ in sorted(enumerate(np.sum(vec, axis=1)), key=lambda x: x[1], reverse=True)]
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
    
    res = [(f"{doc_ids[i]}", corpus_d.id_to_title[doc_ids[i]]) for i, _ in sorted(enumerate(np.sum(vec, axis=1)), key=lambda x: x[1], reverse=True)]
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
    print("1")
    bucket = client.bucket(BUCKET_NAME)
    print("2")
    blob = bucket.blob("page_views/page_views.pkl")
    print("3")
    with blob.open("rb") as f:
        print("1")
        pv_file = pickle.load(f)
    print("4")
    for idd in wiki_ids:
        res.append(pv_file[idd])
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
