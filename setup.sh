#!/bin/bash

# Set global variables
# --------------------

HEADER='\033[38;2;240;128;128m'
MESSAGE='\033[38;2;248;173;157m'
NC='\033[0m'

# Install requirements
# --------------------

printf "${MESSAGE}Installing requirements...${NC}\n"
pip3 install -r requirements.txt > /dev/null 2>&1
printf "${MESSAGE}All requirements have successfully been installed!${NC}\n"
printf "${HEADER}Program is ready to be trained and then predict!${NC}\n"