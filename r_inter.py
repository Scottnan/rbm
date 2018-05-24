import warnings
import seaborn as sns
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, integrate
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2 import robjects
# pandas2ri.activate()
warnings.filterwarnings("ignore")
robjects.r.source('r_source.R')

# 注意：因子为日频，quote为1min线
def get_tf_factor(var, from_to, value_col="IMPUTED"):
    r_var = r['as.character'](robjects.FactorVector(var))
    r_from_to = robjects.IntVector(from_to)
    data = r['tf_factor_tbl'](r['as.character'](r_var), r_from_to, value_col)
    data = pandas2ri.ri2py_dataframe(data)
    print(var)
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


def get_quote_data():
    pass


if __name__ == "__main__":

    path = 'E:/GCAMCDL_DC'
    dirs = os.listdir(path)
    '''
    # ax = sns.distplot(get_tf_factor(["alpha001"], [20170101, 20180518])["alpha001"].tolist(), kde=True, rug=True)
    # plt.show()
    fail = []
    for alpha in dirs:
        print(get_tf_factor([alpha], [20170101, 20180518])[alpha].tolist())
        try:
            ax = sns.distplot(get_tf_factor([alpha], [20170101, 20180518])[alpha].tolist(), kde=True)
            fig = ax.get_figure()
            fig.savefig("plot/" + alpha + ".png")
            plt.close('all')
        except ValueError:
            fail.append(alpha)
            continue
            '''

    d = get_factor_tbl(dirs, [20170101, 20180101])
    d.to_hdf("20180101_20180518.h5", key="DATE")
    print(d)
