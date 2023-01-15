BUCKET_NAME="ex3ir205557564"

#getting the indexes
gsutil cp gs://$BUCKET_NAME/title_postings_gcp/title_index.pkl ~/title_index.pkl
gsutil cp gs://$BUCKET_NAME/anchor_text_postings_gcp/anchor_text_index.pkl ~/anchor_text_index.pkl
gsutil cp gs://$BUCKET_NAME/body_postings_gcp/body_index.pkl ~/body_index.pkl
gsutil cp gs://$BUCKET_NAME/dl_test/lengths.bin ~/dl_lengths.bin
gsutil cp gs://$BUCKET_NAME/dl_test/wiki_ids.bin ~/dl_wiki_ids.bin