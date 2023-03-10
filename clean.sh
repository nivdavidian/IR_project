INSTANCE_NAME="instance1"
REGION=us-central1
ZONE=us-central1-c
PROJECT_NAME="ex3ir-371721"
IP_NAME="$PROJECT_NAME-ip"
GOOGLE_ACCOUNT_NAME="nivdavid" # without the @post.bgu.ac.il or @gmail.com part

# Clean up commands to undo the above set up and avoid unnecessary charges
gcloud compute instances delete -q $INSTANCE_NAME --zone $ZONE
# make sure there are no lingering instances
gcloud compute instances list
# delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080
# delete external addresses
gcloud compute addresses delete -q $IP_NAME --region $REGION