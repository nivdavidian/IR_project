BUCKET_NAME="ex3ir205557564"


gsutil cp gs://$BUCKET_NAME/pr/pr.csv.gz ~/pr.csv.gz
rm -f pr.csv
gunzip pr.csv.gz