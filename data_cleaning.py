#%%
import numpy as np
import pandas as pd
from PIL import Image
import os

class DataCleaning:
    """
    A class for cleaning data.

    Attributes
    -----------
    df : pd.DataFrame

    Methods
    -------
    nan_count(self):
        shows how many null entries are in each columns 

    convert_price(price_column):
        removes £ from price and turns into float

    clean_product_table(price_column):
        cleans the tabular product data
    """
    def __init__(self,dataframe: pd.DataFrame):
        """
        Initialises the object

        Parameters
        ----------
        dataframe: pd.DataFrame
            the dataframe to clean
        """
        self.df = dataframe
    
    def nan_count(self):
        """
        shows how many null values are in each column of a DataFrame

        Returns
        -------
        pd.Series
        """
        return self.df.isnull().sum()
    
    def convert_price(self,price_column):
        """
        removes £ from price and turns into float

        Parameters
        ----------
        price_column:
            name of column to transform
        Returns
        -------
        pd.DataFrame
        """
        df = self.df.copy(deep=True)
        df[price_column] = df[price_column].apply(str)
        df[price_column] = df[price_column].str.replace('£','')
        df[price_column] = df[price_column].str.replace(',','')
        df[price_column] = pd.to_numeric(df[price_column], errors = 'coerce')
        return df
    
    def clean_product_table(self,price_column='price'):
        """
        cleans the product table by applying convert_price and clearing null values
        
        Parameters
        ----------
        price_column:
            name of column to transform, default is 'price'

        Returns
        -------
        pd.DataFrame
        """
        df = self.df.copy(deep=True)
        print(f'NUMBER OF ROWS REMAINING: {len(df)}')
        print('\nCleaning prices...')
        df = DataCleaning(df).convert_price(price_column)
        print(f'NUMBER OF ROWS REMAINING: {len(df)}')
        print('\nCleaning nulls...')
        df = df.dropna(how='any')
        print(f'NUMBER OF ROWS REMAINING: {len(df)}')
        print('\nDisplaying nulls in each column:')
        print(DataCleaning(df).nan_count())
        return df

    

