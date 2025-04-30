#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y python3-pip python3-venv nginx

# Create directories for logs
sudo mkdir -p /var/log/gunicorn
sudo chown -R ubuntu:ubuntu /var/log/gunicorn

# Set up the application directory
mkdir -p ~/rag-api
cd ~/rag-api

# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

# Enable and configure nginx
sudo ln -s /etc/nginx/sites-available/rag-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Start the Gunicorn service
sudo systemctl enable rag-api
sudo systemctl start rag-api

# Check service status
sudo systemctl status rag-api