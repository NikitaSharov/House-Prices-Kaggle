import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

l_features = []
cat = ['GarageType', 'GarageFinish', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1',
       'GarageCond', 'GarageQual', 'BsmtCond', 'BsmtQual', 'FireplaceQu', 'Fence', "KitchenQual",
       "HeatingQC", 'ExterQual', 'ExterCond']
cols = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional"]
ord_col = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond',
           'FireplaceQu']
ordinal_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
fintype_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
fin_col = ['BsmtFinType1', 'BsmtFinType2']
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
column_bin = ['MasVnrArea', 'TotalBsmtFin', 'TotalBsmtSF', '2ndFlrSF', 'WoodDeckSF', 'TotalPorch']
cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]


class Pipeline:

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.df = pd.concat([train, test], ignore_index=True)
        self.train = train
        self.test = test
        self.y = train['SalePrice']

    def prepare_data(self):
        """
         Удаление колонок с большим числом пропущенных значений, заполнение
         пропущенных значений, стандартизация

         Return: train, test DataFrames
         """
        self.df.drop(columns=['Id'], inplace=True)
        quantitative = list(self.df.dtypes[(self.df.dtypes.values == 'float64')
                                           | (self.df.dtypes.values == 'int64')].index)
        quantitative.pop()
        self.df.drop(['GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars', 'SalePrice'], axis=1, inplace=True)
        self.df.drop(['MoSold', 'YrSold'], axis=1, inplace=True)
        list(map(self.get_frequency, self.df.columns))
        self.df.drop(columns=l_features, inplace=True)
        self.df[cat] = self.df[cat].fillna("NA")
        self.df['MSSubClass'] = self.df['MSSubClass'].apply(str)
        self.df[ord_col] = self.df[ord_col].apply(lambda x: x.map(ordinal_map))
        self.df[fin_col] = self.df[fin_col].apply(lambda x: x.map(fintype_map))
        self.df['BsmtExposure'] = self.df['BsmtExposure'].map(expose_map)
        self.df['Fence'] = self.df['Fence'].map(fence_map)
        self.y = np.log(self.y)
        self.df.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)
        columns = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional"]
        self.train = self.df[:self.train.shape[0]]
        self.test = self.df[self.train.shape[0]:]
        for frame in [self.train, self.test]:
            frame[columns] = frame.groupby("Neighborhood")[columns].transform(lambda x: x.fillna(x.mode()[0]))
            frame['LotFrontage'] = frame.groupby('Neighborhood')['LotFrontage'].transform(
                lambda df: df.fillna(df.mean()))
            frame['GarageArea'] = frame.groupby('Neighborhood')['GarageArea'].transform(
                lambda df: df.fillna(df.mean()))
            frame['MSZoning'] = frame.groupby('MSSubClass')['MSZoning'].transform(
                lambda df: df.fillna(df.mode()[0]))
            frame[cont] = frame[cont].fillna(frame[cont].mean())
            frame['TotalLot'] = frame['LotFrontage'] + frame['LotArea']
            frame['TotalBsmtFin'] = frame['BsmtFinSF1'] + frame['BsmtFinSF2']
            frame['TotalSF'] = frame['TotalBsmtSF'] + frame['2ndFlrSF']
            frame['TotalBath'] = frame['FullBath'] + frame['HalfBath']
            frame['TotalPorch'] = frame['OpenPorchSF'] + frame['EnclosedPorch'] + frame['ScreenPorch']
            for column in column_bin:
                bin_name = column + '_bin'
                frame[bin_name] = frame[column].apply(lambda x: 1 if x > 0 else 0)
        self.df = pd.concat([self.train, self.test])
        self.df = pd.get_dummies(self.df)
        self.train = self.df[:self.train.shape[0]]
        self.test = self.df[self.train.shape[0]:]
        numeric = self.train.select_dtypes(np.number).columns
        transformer = RobustScaler().fit(self.train[numeric])
        self.train[numeric] = transformer.transform(self.train[numeric])
        self.test[numeric] = transformer.transform(self.test[numeric])
        return self.train, self.y, self.test

    def get_frequency(self, feature):
        """
        Find columns with  just one value

        Return: value frequency, int
        """
        if self.df[feature].value_counts().apply(lambda x: x / self.df.shape[0]).head(1).values > 0.96:
            l_features.append(feature)
