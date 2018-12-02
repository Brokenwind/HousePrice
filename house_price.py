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


if __name__ == '__main__':
    train, test = load_data()
    train['SalePrice'].describe()