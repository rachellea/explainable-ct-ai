#equality_checks.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import numpy as np
import pandas as pd

#Function for testing equality of arrays
def arrays_equal(output, correct, tol =  1e-6):
    """Check if arrays are equal within tolerance <tol>
    Note that if <tol>==0 then check that arrays are identical.
    Because the following stuff doesn't work at all:
    numpy.testing.assert_almost_equal 
    np.all
    np.array_equal
    np.isclose"""
    assert output.shape == correct.shape
    max_difference = np.amax(np.absolute(output - correct))
    if tol == 0:
        assert max_difference == 0
    else:
        assert max_difference < tol
    return True

#Function for testing equality of dataframes with numeric elements
def dfs_equal(df1, df2):
    assert arrays_equal(df1.values, df2.values, tol = 0)
    assert df1.columns.values.tolist()==df2.columns.values.tolist()
    assert df1.index.values.tolist()==df2.index.values.tolist()
    return True

#Function for testing exact equality of dataframes (e.g. with string elements)
def dfs_str_equal(df1, df2):
    assert (df1.values==df2.values).all()
    assert df1.columns.values.tolist()==df2.columns.values.tolist()
    assert df1.index.values.tolist()==df2.index.values.tolist()
    return True

#Function for testing equality of dataframes with numeric or object elements
def dfs_equal_by_type(df1, df2, numeric_cols, object_cols):
    """<numeric_cols> is a list of strings indicating the columns with numeric
    values. <object_cols> is a list of strings indicating the columns with
    object values."""
    assert arrays_equal(df1[numeric_cols].values, df2[numeric_cols].values, tol = 0)
    assert (df1[object_cols].values==df2[object_cols].values).all()
    assert df1.columns.values.tolist()==df2.columns.values.tolist()
    assert df1.index.values.tolist()==df2.index.values.tolist()
    return True
