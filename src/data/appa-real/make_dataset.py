# -*- coding: utf-8 -*-
from pathlib import Path
import os
import pandas as pd
import subprocess

def _create_anotation(data_dir, data_type):
    assert(data_type in ("train", "valid", "test"))
    csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
    img_dir = Path(data_dir).joinpath(data_type)
    ex_path = f"allcategories_{data_type}.csv"
    
    paths = []
    ages = []
    genders = []
    
    df = pd.read_csv(str(csv_path))
    ex_df = pd.read_csv(str(ex_path))
    
    ignore_path = "ignore_list.csv"
    ignore_img_names = list(pd.read_csv(ignore_path)["img_name"].values)

    for index, row in df.iterrows():
        img_name = row["file_name"]

        if img_name in ignore_img_names:
            continue

        img_path = img_dir.joinpath(img_name + "_face.jpg")
        assert(img_path.is_file())
        paths.append(str(img_path))
        ages.append(row["apparent_age_avg"])
        genders.append(int(ex_df["gender"][index] == "male"))
        
    columns = {}
    columns["file_name"] = paths
    columns["age"] = ages
    columns["gender"] = genders
    data = list(zip(columns["file_name"], columns["age"], columns["gender"]))
    df = pd.DataFrame(data = data)
    df.to_csv(f"/home/Data/appa-real/processed/{data_type}.csv", index=False, header=["file_name", "age", "gender"])
    
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # get the dataset
    subprocess.call(["bash", "get_dataset.sh"])

    # create anotation file : train.csv, valid.csv, test.csv
    data_dir = "/home/Data/appa-real/interim/appa-real-release/"
    for data_type in ("train", "valid", "test"):
        _create_anotation(data_dir, data_type)

if __name__ == '__main__':
    main()
