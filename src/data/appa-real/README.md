# APPA-REAL Dataset

Use [APPA-REAL Dataset](http://chalearnlap.cvc.uab.es/dataset/26/description/) for finetuning CNNs for age estimation.

> The APPA-REAL database contains 7,591 images with associated real and apparent age labels. The total number of apparent votes is around 250,000. On average we have around 38 votes per each image and this makes the average apparent age very stable (0.3 standard error of the mean).

## Clean and prepare dataset

1. Move to `src/data/appa-real/` directory.
2. Run the command below to prepare the dataset: 

```bash
python make_dataset.py
```


## Ignored List
The file named [ignored list](ignore_list.txt), from [this great works](https://github.com/yu4u/age-gender-estimation/tree/master/appa-real), is used to exclude the inappropriate images (only for training set).


