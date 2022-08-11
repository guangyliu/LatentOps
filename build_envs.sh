
# Install required pip packages
pip install transformers==4.19.0 
pip install nltk==3.7 
pip install boto3 sacremoses tensorboardX torchdiffeq

# Install Apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..

