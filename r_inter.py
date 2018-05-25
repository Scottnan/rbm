import warnings
import os
import gc
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2 import robjects
warnings.filterwarnings("ignore")
robjects.r.source('r_source.R')


def get_tf_factor(var, from_to, value_col="IMPUTED"):
    r_var = r['as.character'](robjects.FactorVector(var))
    r_from_to = robjects.IntVector(from_to)
    data = r['tf_factor_tbl'](r['as.character'](r_var), r_from_to, value_col)
    data = pandas2ri.ri2py_dataframe(data)
    print(var[0])
    gc.collect()
    return data


def get_factor_tbl(factors, from_to, value_col="IMPUTED"):
    value = map(lambda factor: get_tf_factor([factor], from_to, value_col), factors)
    data = next(value)
    while True:
        try:
            a = next(value)
            data = pd.merge(data, a)
        except StopIteration:
            break
    return data


if __name__ == "__main__":

    path = 'E:/GCAMCDL_DC'
    dirs = os.listdir(path)
    d = get_factor_tbl(dirs, [20100101, 20111231])
    d.to_hdf("20100101_20111231.h5", key="DATE")
    print(d)
