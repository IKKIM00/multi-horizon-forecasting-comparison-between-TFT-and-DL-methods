# General Description

Updated version for stock, pm2.5 datasets multi-horizon forecasting using Temporal Fusion Transformers(TFT).

You can find TFT paper through this paper link(https://arxiv.org/pdf/1912.09363.pdf).

Used stock, beijing pm2.5 and energy transformer(ET) dataset for multi-horizon forecasting.

# Download Dataset

You can download each dataset through below URLs.

## Stock Dataset
https://finance.yahoo.com/quote/CSV?p=CSV&.tsrc=fin-srch

Also you can use the data uploaded as `data/stock_data.csv`

## Beijing pm2.5 dataset
You can download it through UCI Machine Learning Repository.

https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

Also you can use the data uploaded as `data/beijing_data.csv`

## ET dataset

You can download the dataset through this github link: https://github.com/zhouhaoyi/ETDataset

# How to use
## For Deep Learning(DL) Methods
Through each `ipynb` files, you can choose model you want to use. 

## For TFT Methods
Through `ipynb` files that have `tft` in the file name, you can see the whole procedure. 

Before running cells, make sure to clone original tft github page: https://github.com/google-research/google-research/tree/master/tft. 
