#!/bin/bash

# Copy relevant files to all nodes listed in pscp_hosts.txt
prsync -h pscp_hosts.txt -r . `pwd`

