from google.cloud import storage
import pickle
from inverted_index_gcp import read_dl

BUCKET_NAME = "ex3ir205557564"

class CorpusData:
    def __init__(self, n, id_to_title):
        self.N = n
        self.id_to_title = id_to_title
        d1 = read_dl("dl_test/lengths.bin", n,7,1)
        d2 = read_dl("dl_test/wiki_ids.bin", n,15,2)
        self.dl = {doc_id:length for doc_id, length in map(lambda x,y: (x,y), d1, d2)}
        del d1
        del d2
        
    def write_to_blob(self, path, file_name):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{path}/{file_name}")
        with blob.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def read_from_blob(path, file_name):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{path}/{file_name}")
        with blob.open("rb") as f:
            return pickle.load(f)