import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime as dt
from tqdm import tqdm

import torch

def preprocess(file):
    data = pd.read_csv(file, index_col=0)

    columns = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    scaler = StandardScaler()
    transformed_data = data[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']]
    transformed_data = scaler.fit_transform(transformed_data)
    transformed_data = pd.DataFrame(transformed_data, columns=columns)

    le = LabelEncoder()
    cbwd = data['cbwd']
    transformed_cbwd = le.fit_transform(cbwd)
    transformed_data['cbwd'] = transformed_cbwd

    date_list = list()
    for i in range(len(data)):
        year = int(data.iloc[i]['year'])
        month = int(data.iloc[i]['month'])
        day = int(data.iloc[i]['day'])
        hour = int(data.iloc[i]['hour'])
        date_list.append(dt.datetime(year, month, day, hour))
    transformed_data.index = date_list
    return transformed_data

def data_split(transformed_data):
    train_start = dt.datetime(2010, 1, 1, 0)
    train_end = dt.datetime(2012, 12, 31, 23)
    train = transformed_data.loc[train_start: train_end]

    valid_start = dt.datetime(2013, 1, 1, 0)
    valid_end = dt.datetime(2013, 12, 31, 23)
    valid = transformed_data[valid_start: valid_end]

    test_start = dt.datetime(2014, 1, 1, 0)
    test_end = dt.datetime(2014, 12, 31, 23)
    test = transformed_data[test_start:test_end]
    return train, valid, test

def data2tensor(train, valid, test):

    def split_by_seq(transformed_df):
        X_data, y_data = list(), list()
        for i in tqdm(range(0, len(transformed_df) - 36)):
            pm = transformed_df.iloc[i: i + 24]['pm2.5']
            dewp = transformed_df.iloc[i: i + 24]['DEWP']
            temp = transformed_df.iloc[i: i + 24]['TEMP']
            pres = transformed_df.iloc[i: i + 24]['PRES']
            lws = transformed_df.iloc[i: i + 24]['Iws']
            cbwd = transformed_df.iloc[i: i + 24]['cbwd']
            Is = transformed_df.iloc[i: i + 24]['Is']
            Ir = transformed_df.iloc[i: i + 24]['Ir']
            X_data.append([pm, dewp, temp, pres, lws, cbwd, Is, Ir])

            pm = transformed_df.iloc[i + 24]['pm2.5']
            y_data.append(pm)
        return torch.Tensor(X_data), torch.Tensor(y_data)

    X_train, y_train = split_by_seq(train)
    X_valid, y_valid = split_by_seq(valid)
    X_test, y_test = split_by_seq(test)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)