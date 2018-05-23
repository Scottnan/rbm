from rpy2.robjects import r
from rpy2 import robjects
from sklearn import preprocessing
import pandas as pd
import numpy as np
# pandas2ri.activate()
import warnings
warnings.filterwarnings("ignore")


# Caution: All the date are string form.
# This class is used to get return series of stocks.
class DataHandler(object):
    def __init__(self, date, stock_pool=None, daily_lag=20, month_lag=12):
        r['library']('GCAMCTS')
        self.date = date
        # self.r_from_to = robjects.IntVector(from_to)
        # self.dates_daily, self.dates_month, self.dates_sample = self._get_trading_date()
        # self.daily_lag = daily_lag
        # self.month_lag = month_lag
        # self.stock_pool = stock_pool
        self.data = self._get_quote()
            
    def _get_quote(self):
        r_script = '''
                      data <- read_hfd("{}")
                      
                   '''.format(self.date)
        data = r(r_script)
        return data
        
    @property
    def _daily_rtn_series(self):
        df = pd.concat([self.data[['INNERCODE', 'TRADINGDAY']], 
                        self.data.ix[:,'PREVCLOSEPRICE'] / self.data.ix[:,'CLOSEPRICE'] - 1],
                        axis=1)
        df = df[df['INNERCODE'] > 0]
        df = df.set_index(['INNERCODE', 'TRADINGDAY']).unstack()
        df.columns = df.columns.levels[1].values
        return df
    
    @property
    def _monthly_rtn_series(self):
        t = self.dates_sample[0]
        period = list(set(self._get_pre_months(t, self.month_lag+2) + self.dates_sample))
        df = self.data[self.data['TRADINGDAY'].isin(period)]
        df = pd.concat([df[['INNERCODE', 'TRADINGDAY']], 
                        df.ix[:,'PREVCLOSEPRICE'] / df.ix[:,'CLOSEPRICE'] - 1],
                        axis=1)
        df = df[df['INNERCODE'] > 0]
        df = df.set_index(['INNERCODE', 'TRADINGDAY']).unstack()
        df.columns = df.columns.levels[1].values
        return df
    
    def _get_trading_date(self, start=20050801):
        # This can be monthly, weekly or daily(i choose monthly for convenience)
        # The trading date(sample date) is the last dealing day every month
        # However, i will overweite this function to choose trading date 
        dates_sample = r['as.character']((r['factor_dates'](self.r_from_to, "monthly")))
        r_from_to = robjects.IntVector([start, self.from_to[1]])
        dates_daily = r['as.character']((r['factor_dates'](r_from_to, "daily")))
        dates_month = r['as.character']((r['factor_dates'](r_from_to, "monthly")))
        return dates_daily, dates_month, dates_sample
    
    def get_input_data(self):
        # Some preprocessing and reshape for raw data(see Lawrence paper)
        _monthly_rtn_series = self._monthly_rtn_series
        columns = ["m-%s" % (n+2) for n in range(self.month_lag)] + ["d-%s" % (n+1) for n in range(self.daily_lag)]
        columns = columns+['JAN', 't+1', 'date']
        df = pd.DataFrame(columns=columns)
        for t in self.dates_sample:
            print(t)
            month_date = [i for i in self._get_pre_months(t, self.month_lag+2)[:-2]]  # drop the last month rtn
            daily_date = [i for i in self._get_pre_days(t, self.daily_lag)]
            try:
                t_plus = _monthly_rtn_series[self._get_next_month(t)]
            except IndexError:
                continue
            cumret_daily = (self._daily_rtn_series[daily_date] + 1).cumprod(axis=1).dropna()
            cumret_month = (self._monthly_rtn_series[month_date] + 1).cumprod(axis=1).dropna()
            scaler_daily = preprocessing.StandardScaler().fit(cumret_daily)
            scaler_month = preprocessing.StandardScaler().fit(cumret_month)
            zscore_daily = pd.DataFrame(scaler_daily.transform(cumret_daily),
                                        index=cumret_daily.index, columns=daily_date)
            zscore_month = pd.DataFrame(scaler_month.transform(cumret_month),
                                        index=cumret_month.index, columns=month_date)
            data = pd.concat([zscore_month, zscore_daily], axis=1)
            if t[5:7] == '12':
                data['JAN'] = 1
            else:
                data['JAN'] = 0
            data = pd.concat([data, t_plus], axis=1)
            data['date'] = t
            data.columns = columns
            data = data[data['t+1'].notnull()]
            data.ix[data['t+1'].rank() >= 0.7*max(data['t+1'].rank()), 'label'] = 1
            data.ix[data['t+1'].rank() < 0.3*max(data['t+1'].rank()), 'label'] = 0
            data.drop(['t+1'], axis=1, inplace=True)
            data.rename(columns={'label': 't+1'}, inplace=True)
            data.dropna(inplace=True)
            df = pd.concat([df, data])
            data.to_excel(str(t)+'.xlsx')
        df.to_excel('data.xlsx')
        return df
        
    def _get_next_month(self, t):
        index = self.dates_month.index(t)
        return self.dates_month[index+1]
        
    def _get_pre_months(self, t, length):
        index = self.dates_month.index(t)
        p_i = index-length+1
        return self.dates_month[p_i:index+1]
    
    def _get_pre_days(self, t, length):
        index = self.dates_daily.index(t)
        p_i = index-length+1
        return self.dates_daily[p_i:index+1]

    # "CSI300", "CSI500", "CSI800"
    def _set_stock_pool(self, start, raw_data):
        r_script = '''from_to <- c({}, {})
                      data <- index_comp_wt("{}", from_to = from_to, freq="daily")[, .(INNER_CODE, DATE)]
                      data[,DATE := as.character(DATE)]
                   '''.format(start, self.from_to[1], self.stock_pool)
        pool = r(r_script)
        pool.columns = ['INNERCODE', 'TRADINGDAY']
        return pd.merge(pool, raw_data)


class DataReader(object):
    def __init__(self, path):
        self.data = pd.read_excel(path)
        self.rtn, self.label, self.date = np.array(self.data.ix[:, :-2]), np.array(self.data.ix[:, -2]),\
                                          np.array(self.data.ix[:, -1])
        self.i = 0        
        
    def next_batch(self, batchsize):
        self.iterations = int(len(self.data) / batchsize)
        c = len(self.data.columns)-1
        batch_xs = self.data.iloc[self.i*batchsize:(self.i+1)*batchsize, :c]
        betch_ys = self.data.iloc[self.i*batchsize:(self.i+1)*batchsize, c]
        self.i += 1
        self.i = self.i % batchsize
        return np.array(batch_xs), np.array(betch_ys)


if __name__ == "__main__":
    dh = DataHandler([20180103])
    dh.get_input_data()
    '''
    dr = DataReader('./CSI500/test/data.xlsx')
    print(dr.rtn.shape)
    '''

