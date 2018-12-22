import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print(train.info())
    print('-'*40)
    print(test.info())

    return train, test

def count_missing_data(data):
    null_total = train.isnull().sum(axis=0).sort_values(ascending=False)
    null_rate = null_total / len(train)
    missing_data = pd.concat([null_total,null_rate],axis=1,keys=['Total','Rate'])
    print(missing_data)

    return missing_data


def check_pool_qc(data):
    show_cols = ['PoolQC','SalePrice']
    print(train[~train['PoolQC'].isnull()][show_cols])

def check_misc_feature(data):
    show_cols = ['MiscFeature','SalePrice']
    misc_count = data.groupby('MiscFeature')['MiscFeature'].count()
    #print(train[~train['MiscFeature'].isnull()][show_cols])
    print(misc_count)

def check_alley(data):
    show_cols = ['Alley','SalePrice']
    misc_count = data.groupby('Alley')['Alley'].count()
    #print(train[~train['MiscFeature'].isnull()][show_cols])
    print(misc_count)

def check_electrical(data):
    show_cols = ['Electrical','SalePrice']
    electrical_count = data.groupby('Electrical')['Electrical'].count()
    #print(train[~train['MiscFeature'].isnull()][show_cols])
    print(electrical_count)

def check_mszoning(data):
    show_cols = ['MSZoning','SalePrice']
    ms_count = data.groupby('MSZoning')['MSZoning'].count()
    print(ms_count)
    #print(train[~train['MSZoning'].isnull()][show_cols])

def saleprice_out_liar(data):
    var = 'SalePrice'
    price = data[var][:,np.newaxis]
    ss = StandardScaler().fit(price)
    price_scale = ss.transform(price)
    price_scale.sort(axis=0)
    print('outer range (low) of the distribution:')
    print(price_scale[:10])
    print('-'*40)
    print('outer range (low) of the distribution:')
    print(price_scale[-10:])

def check_sale_price(data):
    var = 'SalePrice'
    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
    sns.distplot(data[var], fit=norm, ax=axes[0][0])
    stats.probplot(data[var],plot=axes[0][1])
    #
    log_price = np.log(data[var])
    sns.distplot(log_price, fit=norm, ax=axes[1][0])
    stats.probplot(log_price,plot=axes[1][1])

    plt.show()



if __name__ == '__main__':
    train, test = load_data()
    cols = train.columns.tolist()
    print(train.groupby(['SaleType'])['SaleType'].count())
    object_cols = []
    numerical_cols = []
    for col in cols:
        if train[col].dtype == 'object':
            object_cols.append(col)
        else:
            numerical_cols.append(col)
    print(object_cols)
    print(numerical_cols)
    #count_missing_data(train)
    #check_mszoning(train)
    #check_misc_feature(train)
    #check_alley(train)
    check_electrical(train)
