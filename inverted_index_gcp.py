from collections import Counter
import itertools
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict


# Let's start with a small block size of 30 bytes just to test things out. 
BLOCK_SIZE = 4*(10**6) # 4Mb
BUCKET_NAME = "ex3ir205557564"

def write_dl(path, dl, bit_limit, num_bytes):
    MASK = 2**bit_limit-1 #mask last bit of the 2 byte(8 bits)
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)
    mask_15 = 2**bit_limit-1
    bit_16 = 2**bit_limit
    with blob.open("wb") as f:
        for d in dl:
            hmbiw = 0
            bit_length = d.bit_length()
            while(hmbiw + bit_limit < bit_length):
                f.write(((d&mask_15)|bit_16).to_bytes(num_bytes, 'big'))
                hmbiw += bit_limit
                d = d >> bit_limit
            f.write((d&mask_15).to_bytes(num_bytes,'big'))

def read_dl(path, length, bit_limit, num_bytes):
    MASK = 2**bit_limit-1 #mask last bit of the 2 byte(8 bits)
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)
    dl = []
    bs= []
    with blob.open("rb") as f:
        bs = f.read()
    index = 0
    n = length
    while(n>0):
        bytess = []
        d=0
        hmi = 0
        while(int.from_bytes(bs[index:index+num_bytes], 'big') > MASK):
            d += (int.from_bytes(bs[index:index+num_bytes], 'big')&MASK)<<hmi
            index += num_bytes
            hmi+=bit_limit
        d += (int.from_bytes(bs[index:index+num_bytes], 'big')&MASK)<<hmi
        dl.append(d)
        n -= 1
        index+=num_bytes
    
    storage_client.close()
    del storage_client
    return dl

def read_from_bucket(folder_name, locs, n_bytes):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    b = []
    for f_name, offset in locs:
        blob = bucket.blob(f"{folder_name}/{f_name}")
        with blob.open("rb") as f:
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
    storage_client.close()
    del storage_client
    return b''.join(b)

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name, bucket_name, folder_prefix):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb') 
                          for i in itertools.count())
        self._f = next(self._file_gen)
        # Connecting to google storage bucket. 
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self._folder_prefix = folder_prefix
        
    
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
        # if the current file is full, close and open a new one.
            if remaining == 0:  
                self._f.close()
                self.upload_to_gcp()                
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()
    
    def upload_to_gcp(self):
        '''
            The function saves the posting files into the right bucket in google storage.
        '''
        file_name = self._f.name
        blob = self.bucket.blob(f"{self._folder_prefix}_postings_gcp/{file_name}")
        blob.upload_from_filename(file_name)

        

class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)
  
    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False 


from collections import defaultdict
from contextlib import closing

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


class InvertedIndex:  
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        self.N = 0
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally), 
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of 
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are 
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents 
        # the number of bytes from the beginning of the file where the posting list
        # starts. 
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage 
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state
    
    def read_term_pl(self, w, folder_name):
        locs = self.posting_locs[w]
        b = read_from_bucket(folder_name, locs, self.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(self.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list

    def posting_lists_iter(self, folder_name):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        for w, locs in self.posting_locs.items():
            b = read_from_bucket(folder_name, locs, self.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            yield w, posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()


    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name, name):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        writer = MultiFileWriter(".", bucket_id, bucket_name, name)
        for w, pl in list_w_pl: 
            # convert to bytes
            b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                          for doc_id, tf in pl])
            # write to file(s)
            locs = writer.write(b)
            # save file locations to index
            posting_locs[w].extend(locs)
        writer.close()
        writer.upload_to_gcp() 
        InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name, name)
        return bucket_id

    
    @staticmethod
    def _upload_posting_locs(bucket_id, posting_locs, bucket_name, name):
        with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
            pickle.dump(posting_locs, f)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(f"{name}_postings_gcp/{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")
    

