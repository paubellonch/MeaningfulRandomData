"""Transformers for categorical data."""

import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import norm

from rdt.transformers.base import BaseTransformer

MAPS = {}

class OneHotEncodingTransformer(BaseTransformer):
    #OneHotEncoding para categorical data
    

    dummy_na = None
    dummies = None

    def __init__(self, error_on_unknown=True):
        self.error_on_unknown = error_on_unknown

    @staticmethod
    def _prepare_data(data):
        """Transform data to appropriate format.

        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.

        Args:
            data (pandas.Series, numpy.ndarray, list or list of lists):
                Data to prepare.

        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('Unexpected format.')
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError('Unexpected format.')

            data = data[:, 0]

        return data

    def fit(self, data):
        """Fit the transformer to the data.

        Get the pandas `dummies` which will be used later on for OneHotEncoding.

        Args:
            data (pandas.Series, numpy.ndarray, list or list of lists):
                Data to fit the transformer to.
        """
        data = self._prepare_data(data)
        self.dummy_na = pd.isnull(data).any()
        self.dummies = list(pd.get_dummies(data, dummy_na=self.dummy_na).columns)

    def transform(self, data):
        """Replace each category with the OneHot vectors.

        Args:
            data (pandas.Series, numpy.ndarray, list or list of lists):
                Data to transform.

        Returns:
            numpy.ndarray:
        """
        data = self._prepare_data(data)
        dummies = pd.get_dummies(data, dummy_na=self.dummy_na)
        array = dummies.reindex(columns=self.dummies, fill_value=0).values.astype(int)
        for i, row in enumerate(array):
            if np.all(row == 0) and self.error_on_unknown:
                raise ValueError(f'The value {data[i]} was not seen during the fit stage.')

        return array

    def reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)
        return pd.Series(indices).map(self.dummies.__getitem__)

