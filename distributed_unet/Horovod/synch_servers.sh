#!/bin/bash

# Copy relevant files to all nodes listed in pscp_hosts.txt
pssh -h pscp_hosts.txt mkdir -p `pwd`
prsync -h pscp_hosts.txt -r . `pwd`
