import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime as dt
from tqdm import tqdm

import torch

def beijing_preprocess(file):
    data = pd.read_csv(file, index_col=0)
    data = data.fillna(0)

    le = LabelEncoder()
    cbwd = data['cbwd']
    transformed_cbwd = le.fit_transform(cbwd)
    data['cbwd'] = transformed_cbwd

    date_list = list()
    for i in range(len(data)):
        year = int(data.iloc[i]['year'])
        month = int(data.iloc[i]['month'])
        day = int(data.iloc[i]['day'])
        hour = int(data.iloc[i]['hour'])
        date_list.append(dt.datetime(year, month, day, hour))
    data.index = date_list
    return data

def beijing_data_split(transformed_data):
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

def beijing_data2tensor(train, valid, test):

    def convert2torch(transformed_df):
        X_data, y_data = list(), list()
        for i in tqdm(range(0, len(transformed_df) - 24)):
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

    columns = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    def transform(data):
        transformed_data = data[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']]
        scaler = StandardScaler()
        transformed_data = scaler.fit_transform(transformed_data)
        transformed_data = pd.DataFrame(transformed_data, columns=columns)
        return transformed_data, scaler

    train_transformed, _ = transform(train)
    train_transformed['cbwd'] = train['cbwd'].values

    valid_transformed, _ = transform(valid)
    valid_transformed['cbwd'] = valid['cbwd'].values

    test_transformed, test_scaler = transform(test)
    test_transformed['cbwd'] = test['cbwd'].values

    X_train, y_train = convert2torch(train_transformed)
    X_valid, y_valid = convert2torch(valid_transformed)
    X_test, y_test = convert2torch(test_transformed)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), test_scaler

def stock_preprocess(file):
    data = pd.read_csv(file, index_col=0, parse_dates=True)
    data = data[['Close', 'Open', 'High', 'Low', 'Volume']]
    return data

def stock_data_split(transformed_data):
    train_start = dt.date(1996, 8, 9)
    train_end = dt.date(2011, 12, 31)
    train = transformed_data.loc[train_start:train_end]

    val_start = dt.date(2012, 1, 1)
    val_end = dt.date(2016, 12, 31)
    valid = transformed_data.loc[val_start:val_end]

    test_start = dt.date(2016, 1, 1)
    test_end = dt.date(2020, 4, 7)
    test = transformed_data.loc[test_start:test_end]

    return train, valid, test

def stock_data2tensor(train, valid, test):

    def convert2torch(data, time_step):
        X_data, y_data = list(), list()
        for i in range(time_step, len(data)):
            X_data.append(data.iloc[i - time_step:i].values)
            y_data.append(data.iloc[i][0])
        X_data, y_data = torch.Tensor(X_data), torch.Tensor(y_data)
        return X_data, y_data

    def transform(data):
        columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        scaler = StandardScaler()
        idx = data.index
        transformed_data = data[['Close', 'Open', 'High', 'Low', 'Volume']]
        transformed_data = scaler.fit_transform(transformed_data)
        transformed_data = pd.DataFrame(transformed_data, columns=columns)
        transformed_data.index = idx
        return transformed_data, scaler

    train_transformed, _ = transform(train)
    valid_transformed, _ = transform(valid)
    test_transformed, test_scaler = transform(test)

    X_train, y_train = convert2torch(train_transformed, 50)
    X_valid, y_valid = convert2torch(valid_transformed, 50)
    X_test, y_test = convert2torch(test_transformed, 50)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), test_scaler

def ET_preprocess(data_dir):
    return pd.read_csv(data_dir, parse_dates=True, index_col=0)

def ET_data_split(data):
    train_start = dt.datetime(2016, 7, 1, 0, 0)
    train_end = dt.datetime(2017, 6, 30, 23, 45)

    valid_start = dt.datetime(2017, 7, 1, 0, 0)
    valid_end = dt.datetime(2018, 1, 31, 23, 45)

    test_start = dt.datetime(2018, 2, 1, 0, 0)
    test_end = dt.datetime(2018, 6, 26, 19, 45)

    train = data.loc[train_start: train_end]
    valid = data.loc[valid_start: valid_end]
    test = data.loc[test_start: test_end]
    return train, valid, test

def ET_data2tensor(train, valid, test):

    def convert2torch(transformed_df):
        X_data, y_data = list(), list()
        for i in tqdm(range(0, len(transformed_df) - 20)):
            hufl = transformed_df.iloc[i: i + 20]['HUFL']
            hull = transformed_df.iloc[i: i + 20]['HULL']
            mufl = transformed_df.iloc[i: i + 20]['MUFL']
            mull = transformed_df.iloc[i: i + 20]['MULL']
            lufl = transformed_df.iloc[i: i + 20]['LUFL']
            lull = transformed_df.iloc[i: i + 20]['LULL']

            X_data.append([hufl, hull, mufl, mull, lufl, lull])

            ot = transformed_df.iloc[i + 24]['OT']
            y_data.append(ot)
        return torch.Tensor(X_data), torch.Tensor(y_data)

    columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    def transform(data):
        scaler = StandardScaler()
        transformed_data = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']]
        transformed_data = scaler.fit_transform(transformed_data)
        transformed_data = pd.DataFrame(transformed_data, columns=columns)
        return transformed_data, scaler

    train_transformed, _ = transform(train)
    valid_transformed, _ = transform(valid)
    test_transformed, test_scaler = transform(test)

    X_train, y_train = convert2torch(train_transformed)
    X_valid, y_valid = convert2torch(valid_transformed)
    X_test, y_test = convert2torch(test_transformed)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), test_scaler