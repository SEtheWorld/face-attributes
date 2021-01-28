#!/bin/bash
rawdatadir=/home/Data/apparent_v1/raw/
interdatadir=/home/Data/apparent_v1/interim/
echo "Creating interim folder..."
mkdir -p $interdatadir
echo "Creating raw folder..."
mkdir -p $rawdatadir
echo "Change directory to raw folder"
cd $rawdatadir

data_list=('http://158.109.8.102/ApparentAgeV1/Train.zip' 'http://158.109.8.102/ApparentAgeV1/TrainLabels.zip' 'http://158.109.8.102/ApparentAgeV1/Validation.zip' 'http://158.109.8.102/ApparentAgeV1/ValidationLabels.zip')

echo "Downloading dataset..."
for val in ${data_list[*]};
    do
    readarray -d / -t strarr <<< "$val"
    if [ ! -f "$rawdatadir${strarr[4]}" ]
    then
        wget $val
        unzip ${strarr[4]} -d $rawdatadir
        rm ${strarr[4]}
    fi
    # echo $val
    done
echo "Finish download"
