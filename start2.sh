INSTANCE_NAME="instance1"
REGION=us-central1
ZONE=us-central1-c
PROJECT_NAME="ex3ir-371721"
IP_NAME="$PROJECT_NAME-ip"
GOOGLE_ACCOUNT_NAME="nivdavid" # without the @post.bgu.ac.il or @gmail.com part

# 4. Secure copy your app to the VM
# gcloud compute scp ~/IR_project/search_frontend.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME #Mac
# gcloud compute scp ~/IR_project/inverted_index_gcp.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME #Mac
# gcloud compute scp ~/IR_project/get_indexes.sh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME #Mac

gcloud compute scp ~/Projects/IR_project/search_frontend.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME #Linux
gcloud compute scp ~/Projects/IR_project/inverted_index_gcp.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME #Linux
gcloud compute scp ~/Projects/IR_project/corpus_data.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME #Linux
gcloud compute scp ~/Projects/IR_project/get_indexes.sh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME #Linux

# 5. SSH to your VM and start the app
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME
# python3 search_frontend.py