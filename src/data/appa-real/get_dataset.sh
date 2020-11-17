#!/bin/bash
rawdatadir=/home/Data/appa-real/raw/
interdatadir=/home/Data/appa-real/interim/
mkdir -p $interdatadir
mkdir -p $rawdatadir
cd $rawdatadir
if [ ! -f "/home/Data/appa-real/raw/appa-real-release.zip" ]
then
	wget http://158.109.8.102/AppaRealAge/appa-real-release.zip
	unzip appa-real-release.zip -d $interdatadir
fi
