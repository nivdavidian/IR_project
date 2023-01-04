INSTANCE_NAME="instance-1"
REGION=us-central1
ZONE=us-central1-c
PROJECT_NAME="ex3ir-371721"
IP_NAME="$PROJECT_NAME-ip"
GOOGLE_ACCOUNT_NAME="nivdavid" # without the @post.bgu.ac.il or @gmail.com part

# 0. Install Cloud SDK on your local machine or using Could Shell
# check that you have a proper active account listed
gcloud auth list 
# check that the right project and zone are active
gcloud config list
# if not set them
# gcloud config set project $PROJECT_NAME
# gcloud config set compute/zone $ZONE

# 1. Set up public IP
gcloud compute addresses create $IP_NAME --project=$PROJECT_NAME --region=$REGION
gcloud compute addresses list
# note the IP address printed above, that's your extrenal IP address.
# Enter it here: 
read ip
INSTANCE_IP=$ip

# 2. Create Firewall rule to allow traffic to port 8080 on the instance
gcloud compute firewall-rules create default-allow-http-8080 \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server

# 3. Create the instance. Change to a larger instance (larger than e2-micro) as needed.
gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=e2-micro \
  --network-interface=address=$INSTANCE_IP,network-tier=PREMIUM,subnet=default \
  --metadata-from-file startup-script=startup_script_gcp.sh \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=http-server
# monitor instance creation log using this command. When done (4-5 minutes) terminate using Ctrl+C
gcloud compute instances tail-serial-port-output $INSTANCE_NAME --zone $ZONE