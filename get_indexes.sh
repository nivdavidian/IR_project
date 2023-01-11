BUCKET_NAME="ex3ir205557564"

#getting the indexes
gsutil cp gs://$BUCKET_NAME/title_postings_gcp/title_index.pkl ~/title_index.pkl
gsutil cp gs://$BUCKET_NAME/body_alternate_postings_gcp/body_alternate_index.pkl ~/body_index.pkl