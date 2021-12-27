# use "python -m unittest discover"
# to call the functions

import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import transform as tr


class TestPreProcessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'feature': list(range(10)),
            'target': list(range(10)),
            'Date': pd.date_range('2010-10-01', periods=10)
        })

    def test_create_moving_average_result(self):
        features = self.df.columns.drop('target')
        avg_last_5_feature = [
            np.mean(list(range(5)), dtype='float64'),
            np.mean(list(range(1, 6)), dtype='float64'),
            np.mean(list(range(2, 7)), dtype='float64'),
            np.mean(list(range(3, 8)), dtype='float64'),
            np.mean(list(range(4, 9)), dtype='float64'),
            np.mean(list(range(5, 10)), dtype='float64')
        ]

        test_df = pd.DataFrame({
            'feature': list(range(4, 10)),
            'Date': pd.date_range('2010-10-05', periods=6),
            'avg_last_5_feature': avg_last_5_feature,
            'target': list(range(4, 10))}, index=range(4, 10))

        df = tr.create_moving_average(self.df, features, windows=[5])
        # Drop the first lines that must have null values due to the moving average creation
        assert_frame_equal(df.dropna(), test_df)
