# use "python -m unittest discover"
# to call the functions at the stocks

import unittest
from utils import split_df
import pandas as pd
from pandas.testing import assert_frame_equal


class TestPreProcessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'feature': list(range(10)),
            'target': list(range(10)),
            'Date': pd.date_range('2010-10-01', periods=10)
        })

        self.test_Xtrain = pd.DataFrame({
            'feature': list(range(7)),
            'Date': pd.date_range('2010-10-01', periods=7)
        })
        self.test_Xtest = pd.DataFrame({
            'feature': list(range(7, 10)),
            'Date': pd.date_range('2010-10-08', periods=3)
        }, index=range(7, 10))
        self.test_ytrain = pd.DataFrame({'target': list(range(7))})
        self.test_ytest = pd.DataFrame(
            {'target': list(range(7, 10))}, index=range(7, 10))

    def test_split_df(self):

        Xtrain, Xtest, ytrain, ytest = split_df(
            self.df, target='target', size=0.7)

        assert_frame_equal(Xtrain, self.test_Xtrain)
        assert_frame_equal(Xtest, self.test_Xtest)
        assert_frame_equal(ytrain, self.test_ytrain)
        assert_frame_equal(ytest, self.test_ytest)
