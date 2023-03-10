from google.cloud import storage
import pickle
from inverted_index_gcp import read_dl
import csv

BUCKET_NAME = "ex3ir205557564"

class CorpusData:
    def __init__(self, n, id_to_title, dl=None, pr=None):
        self.N = n
        self.id_to_title = id_to_title
        self.dl = dl
        self.pr = pr
        
    def read_dls(self):
        d1 = read_dl("dl_test/lengths.bin", self.N,7,1)
        d2 = read_dl("dl_test/wiki_ids.bin", self.N,15,2)
        self.dl = {d2[i]:d1[i] for i in range(self.N)}
        del d1
        del d2

    def write_to_blob(self, path, file_name):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{path}/{file_name}")
        with blob.open("wb") as f:
            pickle.dump(self, f)
        
    def read_pr(self):
        if self.pr != None:
            return self.pr
        pr_dict ={}
        with open("/home/nivdavid/pr.csv", "r") as f:
            csv_reader = csv.reader(f)
            data = list(csv_reader)
            pr_dict = dict([(id,rank) for id, rank in data])
        self.pr = pr_dict
        
        return pr_dict
        

    @staticmethod
    def read_from_blob(path, file_name):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{path}/{file_name}")
        with blob.open("rb") as f:
            return pickle.load(f)
