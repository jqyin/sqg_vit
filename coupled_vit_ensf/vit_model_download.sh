#!/bin/bash

download_directory="./coupled_vit_ensf/vit_model"

# Check if the directory already exists, create if it doesn't
if [ ! -d "$download_directory" ]; then
    mkdir -p "$download_directory"
fi
direct_download_url="https://figshare.com/ndownloader/files/45670455?private_link=0f97922cfdf4ec3337fb"


# Use curl to download the file and keep the original filename
cd "$download_directory" # change directory to the desired download location
curl -LJO "$direct_download_url"

echo "Download completed. Check the ${download_directory} directory."
