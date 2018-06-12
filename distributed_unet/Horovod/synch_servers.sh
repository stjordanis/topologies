#!/bin/bash

# Copy relevant files to all nodes listed in pscp_hosts.txt
pscp.pssh -h pscp_hosts.txt ~/unet/single-node/settings.py ~/unet/single-node/
pscp.pssh -h pscp_hosts.txt ~/unet/single-node/hvd_singleworker.sh ~/unet/single-node/
pscp.pssh -h pscp_hosts.txt ~/unet/single-node/hvd_train.py ~/unet/single-node/
pscp.pssh -h pscp_hosts.txt ~/unet/single-node/hvd_multiworker.sh ~/unet/single-node/
