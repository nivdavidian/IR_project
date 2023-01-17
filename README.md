# IR_project
Information retrieval project in BGU course

How to run:

1. Create a cluster and upload in home/dataproc corpus_dat.py, 205557564_gcp.ipynb, inverted_index_gcp.py.
2. Create all the indexes to the bucket u want
3. Create an Instance using command "bash start1.sh" (change necessary variables)
4. Using "bash start2.sh" move necessary files to the instance and then it logs u to the instance
5. After logging to instance run "bash get_indexes.sh"
6. Run "python3 search_frontend.py"