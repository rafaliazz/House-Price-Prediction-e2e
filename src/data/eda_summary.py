"""
File to summarize the main findings from eda.ipynb and creates helper functions for data_preprocessor.py
"""

import pandas as pd
from sklearn.impute import KNNImputer


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df['LotFrontage'] <= 300) &
        (df['LotArea'] <= 100000) &
        (df['BsmtFinSF1'] <= 5000) &
        (df['1stFlrSF'] <= 4000) &
        (df['TotalBsmtSF'] <= 6000) &
        (df['SalePrice'] <= 700000)
    ]


def drop_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])


def fill_categorical_na(df: pd.DataFrame, numerical_features) -> pd.DataFrame:
    for col in [c for c in df.columns if c not in numerical_features]:
        df[col] = df[col].fillna('NA')
    return df


def impute_mas_vnr_area(df: pd.DataFrame) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=4)
    df['MasVnrArea'] = imputer.fit_transform(df[['MasVnrArea']]).round(0)
    return df


def impute_garage_year(df: pd.DataFrame) -> pd.DataFrame:
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    def check_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        return 'Fall'

    df['SeasonSold'] = df['MoSold'].apply(check_season)
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    return df


def add_amenities_features(df: pd.DataFrame) -> pd.DataFrame:
    df['AmenitiesPresence'] = (
        (df['GarageQual'] != 'NA') |
        (df['Fireplaces'] > 0) |
        (df['MiscVal'] > 0)
    ).astype(int)

    df['AmenitiesQuantity'] = (
        (df['GarageQual'] != 'NA').astype(int) +
        (df['Fireplaces'] > 0).astype(int) +
        (df['MiscVal'] > 0).astype(int)
    )
    return df


ORDINAL_FEATURES = [
    'LotShape', 'Utilities', 'LandSlope',
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'Functional',
    'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond'
]

ORDINAL_CATEGORIES = [
    ['IR3', 'IR2', 'IR1', 'Reg'],
    ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
    ['Sev', 'Mod', 'Gtl'],
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['NA', 'No', 'Mn', 'Av', 'Gd'],
    ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['NA', 'Unf', 'RFn', 'Fin'],
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
]

