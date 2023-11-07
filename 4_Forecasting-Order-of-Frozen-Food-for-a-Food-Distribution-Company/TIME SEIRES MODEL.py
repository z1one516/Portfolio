import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
from pycaret.regression import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import optuna
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, MaxPooling1D, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
import optuna.logging as logging


logging.set_verbosity(logging.ERROR) # To control the output during the training process, set the log level.
                                    # For example, the ERROR level produces the least output.

# derived variables 
class Derived_Variable:
    def __init__(self, df, target: str):
        '''
        class to make derived variables 

        Parameters:
        df (dataframe): data
        target (str): name of the target column 

        '''
        self.df = df
        self.target = target

    def log_scale(self, data):
        '''
        log(variable)

        Returns:
        df_log (DataFrame)
        '''

        zero_values = data < 0

        if zero_values.sum() > 0:
            data = data - min(data)
            data = np.log1p(data)
        else:

            data = np.log1p(data)

        return data

    def log_naive_latest_value(self):
        '''
        previous value of log(variable) 
        '''

        self.df['log_naive_latest_value'] = 0
        self.df['log_naive_latest_value'][1:] = self.log_scale(self.df[self.target][:-1])
        self.df['log_naive_latest_value'][0] = None

        return self.df

    def log_diff_1stlag(self):
        '''
        previous value of log(value) differencing 
        '''
        self.df['log_diff_1stlag'] = self.log_scale(self.df[self.target])
        self.df['log_diff_1stlag'][1:] = np.array(self.df['log_diff_1stlag'][1:]) - np.array(
            self.df['log_diff_1stlag'][:-1])
        self.df['log_diff_1stlag'][0] = None
        return self.df

    def diff_1stlag(self):
        '''
        previous value of differenced data 
        '''

        self.df['diff_1stlag'] = 0
        self.df['diff_1stlag'][1:] = np.array(self.df[self.target][1:]) - np.array(self.df[self.target][:-1])
        self.df['log_diff_1stlag'][1:] = self.df['log_diff_1stlag'][:-1]
        self.df['diff_1stlag'][0] = None

        return self.df

    def diff_7weekmean(self):
        '''
        differencing and moving average of 7 weeks 
        '''
        self.df['diff_7weekmean'] = None
        self.df['diff_7weekmean'][1:] = np.array(self.df[self.target][1:]) - np.array(self.df[self.target][:-1])
        self.df['diff_7weekmean'] = self.df['diff_7weekmean'].rolling(window=7).mean().fillna(self.df['diff_7weekmean'])
        self.df['diff_7weekmean'][:6] = None

        return self.df

    def _7weekdiff_1stlag(self):
        '''
        previous value of 7 weeks differencing 
        '''
        self.df['7weekdiff_1stlag'] = None
        self.df['7weekdiff_1stlag'][6:] = np.array(self.df[self.target][6:]) - np.array(self.df[self.target][:-6])
        self.df['7weekdiff_1stlag'] = self.df['7weekdiff_1stlag'].rolling(window=7).mean().fillna(
            self.df['7weekdiff_1stlag'])
        self.df['7weekdiff_1stlag'][:6] = None
        self.df['7weekdiff_1stlag'][1:] = self.df['7weekdiff_1stlag'][:-1]

        return self.df

    def diff_4weekmean(self):
        '''
        differencing and 4 weeks moving average 
        '''
        self.df['diff_4weekmean'] = None
        self.df['diff_4weekmean'][1:] = np.array(self.df[self.target][1:]) - np.array(self.df[self.target][:-1])
        self.df['diff_4weekmean'] = self.df['diff_4weekmean'].rolling(window=4).mean().fillna(self.df['diff_4weekmean'])
        self.df['diff_4weekmean'][:3] = None

        return self.df

    def _4weekdiff_1stlag(self):
        '''
        previous value of 4 weeks differencing 
        '''
        self.df['4weekdiff_1stlag'] = None
        self.df['4weekdiff_1stlag'][3:] = np.array(self.df[self.target][3:]) - np.array(self.df[self.target][:-3])
        self.df['4weekdiff_1stlag'] = self.df['4weekdiff_1stlag'].rolling(window=4).mean().fillna(
            self.df['4weekdiff_1stlag'])
        self.df['4weekdiff_1stlag'][:3] = None
        self.df['4weekdiff_1stlag'][1:] = self.df['4weekdiff_1stlag'][:-1]

        return self.df

    def diff_12weekmean(self):
        '''
        differencing and moving average of 12 weeks 
        '''
        self.df['diff_12weekmean'] = None
        self.df['diff_12weekmean'][1:] = np.array(self.df[self.target][1:]) - np.array(self.df[self.target][:-1])
        self.df['diff_12weekmean'] = self.df['diff_12weekmean'].rolling(window=12).mean().fillna(
            self.df['diff_12weekmean'])
        self.df['diff_12weekmean'][:11] = None

        return self.df

    def _12weekdiff_1stlag(self):
        '''
        12일 차분하고 직전값
        '''
        self.df['12weekdiff_1stlag'] = None
        self.df['12weekdiff_1stlag'][11:] = np.array(self.df[self.target][11:]) - np.array(self.df[self.target][:-11])
        self.df['12weekdiff_1stlag'] = self.df['12weekdiff_1stlag'].rolling(window=12).mean().fillna(
            self.df['12weekdiff_1stlag'])
        self.df['12weekdiff_1stlag'][:11] = None
        self.df['12weekdiff_1stlag'][1:] = self.df['12weekdiff_1stlag'][:-1]

        return self.df

    def region_total_2ndlag(self, period=7):
        '''
        Calculate the 2nd lag value of a rolling mean with a specified 'period' 
        and store it in a new column 'region_total_2ndlag'

        '''
        self.df['region_total_2ndlag'] = None
        self.df['region_total_2ndlag'] = self.df[self.target].rolling(window=period).mean().fillna(self.df[self.target])
        self.df['region_total_2ndlag'][:period] = None
        self.df['region_total_2ndlag'][2:] = self.df['region_total_2ndlag'][:-2]

        return self.df

    def _7weekmax(self):
        '''
       max value in 7 weeks
        '''
        self.df['_7weekmax'] = None
        self.df['_7weekmax'] = self.df[self.target].rolling(window=7).max().fillna(self.df[self.target])
        self.df['_7weekmax'][:6] = None

        return self.df

    def _7weekmin(self):
        '''
        minimum value in 7 weeks
        '''
        self.df['_7weekmin'] = None
        self.df['_7weekmin'] = self.df[self.target].rolling(window=7).min().fillna(self.df[self.target])
        self.df['_7weekmin'][:6] = None

        return self.df

    def _7weekmean(self):
        '''
        average value of 7 weeks
        '''
        self.df['_7weekmean'] = None
        self.df['_7weekmean'] = self.df[self.target].rolling(window=7).mean().fillna(self.df[self.target])
        self.df['_7weekmean'][:6] = None

        return self.df

    def _7weekmedian(self):
        '''
        median value of 7 weeks
        '''
        self.df['_7weekmedian'] = None
        self.df['_7weekmedian'] = self.df[self.target].rolling(window=7).median().fillna(self.df[self.target])
        self.df['_7weekmedian'][:6] = None

        return self.df

    def _7weekstd(self):
        '''
        standard deviation of 7 weeks value 
        '''
        self.df['_7weekstd'] = None
        self.df['_7weekstd'] = self.df[self.target].rolling(window=7).std().fillna(self.df[self.target])
        self.df['_7weekstd'][:6] = None

        return self.df

    def _4weekmax(self):
        '''
        max in 4 weeks 
        '''
        self.df['_4weekmax'] = None
        self.df['_4weekmax'] = self.df[self.target].rolling(window=4).max().fillna(self.df[self.target])
        self.df['_4weekmax'][:3] = None

        return self.df

    def _4weekmin(self):
        '''
        minimum in 4 weeks 
        '''
        self.df['_4weekmin'] = None
        self.df['_4weekmin'] = self.df[self.target].rolling(window=4).min().fillna(self.df[self.target])
        self.df['_4weekmin'][:3] = None

        return self.df

    def _4weekmean(self):
        '''
        average of 4 weeks 
        '''
        self.df['_4weekmean'] = None
        self.df['_4weekmean'] = self.df[self.target].rolling(window=4).mean().fillna(self.df[self.target])
        self.df['_4weekmean'][:3] = None

        return self.df

    def _4weekmedian(self):
        '''
        median of 4 weeks 
        '''
        self.df['_4weekmedian'] = None
        self.df['_4weekmedian'] = self.df[self.target].rolling(window=4).median().fillna(self.df[self.target])
        self.df['_4weekmedian'][:3] = None

        return self.df

    def _4weekstd(self):
        '''
        standard deviation of 4 weeks 
        '''
        self.df['_4weekstd'] = None
        self.df['_4weekstd'] = self.df[self.target].rolling(window=4).std().fillna(self.df[self.target])
        self.df['_4weekstd'][:3] = None

        return self.df

    def _12weekmax(self):
        '''
        max of 12 weeks
        '''
        self.df['_12weekmax'] = None
        self.df['_12weekmax'] = self.df[self.target].rolling(window=12).max().fillna(self.df[self.target])
        self.df['_12weekmax'][:11] = None

        return self.df

    def _12weekmin(self):
        '''
        minimum of 12 weeks 
        '''
        self.df['_12weekmin'] = None
        self.df['_12weekmin'] = self.df[self.target].rolling(window=12).min().fillna(self.df[self.target])
        self.df['_12weekmin'][:11] = None

        return self.df

    def _12weekmean(self):
        '''
        average of 12 weeks 
        '''
        self.df['_12weekmean'] = None
        self.df['_12weekmean'] = self.df[self.target].rolling(window=12).mean().fillna(self.df[self.target])
        self.df['_12weekmean'][:11] = None

        return self.df

    def _12weekmedian(self):
        '''
        median of 12 weeks
        '''
        self.df['_12weekmedian'] = None
        self.df['_12weekmedian'] = self.df[self.target].rolling(window=12).median().fillna(self.df[self.target])
        self.df['_12weekmedian'][:11] = None

        return self.df

    def _12weekstd(self):
        '''
        standard deviation of 12 weeks 
        '''
        self.df['_12weekstd'] = None
        self.df['_12weekstd'] = self.df[self.target].rolling(window=12).std().fillna(self.df[self.target])
        self.df['_12weekstd'][:11] = None

        return self.df

    def all_Variable(self):
        self.df = self.log_naive_latest_value()
        self.df = self.log_diff_1stlag()
        self.df = self.diff_1stlag()
        self.df = self.diff_7weekmean()
        self.df = self._7weekdiff_1stlag()
        self.df = self.diff_4weekmean()
        self.df = self._4weekdiff_1stlag()
        self.df = self.diff_12weekmean()
        self.df = self._12weekdiff_1stlag()
        self.df = self.region_total_2ndlag()
        self.df = self._7weekmax()
        self.df = self._7weekmin()
        self.df = self._7weekmean()
        self.df = self._7weekmedian()
        self.df = self._7weekstd()
        self.df = self._4weekmax()
        self.df = self._4weekmin()
        self.df = self._4weekmean()
        self.df = self._4weekmedian()
        self.df = self._4weekstd()
        self.df = self._12weekmax()
        self.df = self._12weekmin()
        self.df = self._12weekmean()
        self.df = self._12weekmedian()
        self.df = self._12weekstd()

        return self.df

# preprocessing function using z-score 
class Zscore_outlier:
    def __init__(self, df, column: str):
        '''
        Parameters:
        df (dataframe): data
        column (str): column name to preprocess outliers

        '''

        self.org_df = df
        self.df = pd.DataFrame(df[column], columns=[column])
        self.column = column
        self.max_index_list = []
        self.min_index_list = []
        self.max_value = None
        self.min_value = None

    def zscore_interval(self, threshold: float = 3):
        """
        A method that calculates the Z-scores for a specific column of a data frame 
        and returns the indices of data points with Z-scores greater than or equal to the provided threshold.


        Parameters:
        threshold (float, optional): The threshold value for Z-scores (default is 3).

        Returns:
        self.max_index_list, self.min_index_list: 
        Lists that store the indices of data points with Z-scores greater than or equal to the threshold for each colum
        """

        for col in self.df.columns:
            zscore_series = (self.df[col] - self.df[col].mean()) / self.df[col].std()

            self.max_index_list += zscore_series[zscore_series >= threshold].index.tolist()
            self.min_index_list += zscore_series[zscore_series <= (-1 * threshold)].index.tolist()

        return self.max_index_list, self.min_index_list

    def remove_min_max(self):
        """
        In a data frame where outliers have been removed from an arbitrary column, 
        a method to calculate the maximum and minimum values

        Parameters:
        threshold (float, optional): The threshold value for Z-scores (default is 3).
        Returns:
        data_max, data_min: max, minimum
        """

        remove_out = self.df.drop(self.max_index_list + self.min_index_list)

        self.max_value = remove_out[self.column].max()
        self.min_value = remove_out[self.column].min()

        return self.max_value, self.min_value

    def update_values_by_index(self, update_type: str = ["min_max", "ma"]):
        """
        A method to replace the values of rows in a data frame corresponding to the given index list with new values.
        Parameters:
        update_type (str): The type of new values to be inserted.

        Returns:
        DataFrame: A new data frame with the values updated
        """
        new_df = self.org_df.copy()
        self.zscore_interval()
        self.remove_min_max()

        if update_type == 'min_max':
            for i, index in enumerate(self.max_index_list):
                new_df.loc[index, self.column] = self.max_value

            for i, index in enumerate(self.min_index_list):
                new_df.loc[index, self.column] = self.min_value
        elif update_type == 'ma':
            for i, index in enumerate(self.max_index_list):
                new_df.loc[index, self.column] = sum(new_df.loc[index - 14:index, self.column]) / 14

            for i, index in enumerate(self.min_index_list):
                new_df.loc[index, self.column] = sum(new_df.loc[index - 14:index, self.column]) / 14
        else:
            print('error')
            return

        return new_df

# function to fill nan
class Fill_nan:
    def __init__(self, df, columns):
        '''
        Parameters:
        df (dataframe): data to replace nan 
        columns (list): column list to fill nan 
        '''

        self.df = df
        self.columns = columns

    def ma_fill(self, period: int = 7):
        """
        Impute Missing Values with Moving Averages

        Parameters:
        period (int): Moving average period (default is 7).

        Returns:
        df (DataFrame): Data with missing values impute
        """

        # Extract rows with NaN 
        data = self.df.reset_index(drop=True)
        for j in self.columns:
            rows_with_nan = data[data[j].isna()]

            for i in rows_with_nan.index:
                if i - period < 0:
                    data.at[i, j] = data.loc[:i, j].sum() / i

                else:
                    data.at[i, j] = data.loc[i - period:i, j].sum() / period

        return data

    def zero_fill(self):
        """
        fill nan with 0 

        Returns:
        df(DataFrame): data that nan is replaced 
        """

        data = self.df.reset_index(drop=True)
        for j in self.columns:
            rows_with_nan = data[data[j].isna()]

            for i in rows_with_nan.index:
                data.at[i, j] = 0

        return data

    def recent_fill(self):
        """
        fill nan with the recent value 

        Returns:
        df(DataFrame): data that nan is replaced 
        """
        data = self.df.reset_index(drop=True)
        for j in self.columns:
            rows_with_nan = data[data[j].isna()]

            for i in rows_with_nan.index:
                if i - 1 < 0:
                    data.drop(i, axis=0, inplace=True)
                else:
                    data.at[i, j] = data.at[i - 1, j]

        return data

# function to fill missing dates in time series data 
def date_fill(df):
    '''
    a function to fill missing dates in data

    Parameters:
    df (DataFrame): data to fill dates 

    Returns:
    df(DataFrame): data that date has been filled
    '''
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # minimum and max date in dataframe 
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    # create whole data range 
    date_range = pd.date_range(start=min_date, end=max_date)

    # bring missing dates 
    missing_dates = date_range[~date_range.isin(df['Date'])]

    df.set_index('Date', inplace=True)

    for i in missing_dates.tolist():
        df.loc[i] = None

    df.reset_index(inplace=True)
    # sort dataframe based on Date column
    df = df.sort_values(by='Date')
    df.reset_index(drop=True, inplace=True)

    return df

# window, split, scaling 
class Data_preparation:
    def __init__(self, df=None, y: str = None, y_columns: list = None):
        '''
        Parameters:
        df (dataframe): data
        y (str): the time series column to predict 
        y_columns (list): column lists to convert into log 
        '''
        self.df = df
        self.y = y
        self.y_columns = y_columns

    def log_scale(self):
        '''
        log 

        Returns:
        df_log (DataFrame): data log 
        '''
        df_log = self.df.copy()

        for i in self.y_columns:
            zero_values = df_log[i] < 0

            if zero_values.sum() > 0:
                df_log[i] = df_log[i] - min(df_log[i].values)
                df_log[i] = np.log1p(df_log[i])
            else:

                df_log[i] = np.log1p(df_log[i])

        return df_log

    # when logarithm can be taken 

    def sliding_window(self, size: int, term: int = 0, steps: int = 1, scale=[None, 'log']):

        '''
        Function to provide default prediction Dataset

        Parameters:
        size (int): window_size
        term (int): Term at which to predict (default value 0)
        Prediction horizon (default value 1).
        scale : Whether to scale (default value None)

        Returns:
        X_train, y_train, X_test : train and test data

        '''
        df_train = 0

        if scale == 'log':
            df_train = self.log_scale()
            print(df_train)
        else:
            df_train = self.df

        X = []  # input data 
        Y = {}  # target data to predict

        for i in range(len(df_train) - size + 1):
            window_data = df_train[i:i + size][self.y].values # create input data by sliding window 
            X.append(window_data)

        for j in range(steps):
            target_list = [] # Generate target data to predict the values ​​of the next several steps
            for i in range(len(df_train) - size - steps - term + 1):
                target_data = df_train.iloc[i + size + j + term][self.y]
                target_list.append(target_data)
            Y[f'y+{j + 1}'] = target_list

        X = pd.DataFrame(X, columns=[f't-{size - i}' for i in range(size)])

        df_train = pd.concat([df_train.iloc[size:].reset_index(drop=True), X], axis=1)

        df_train.drop(self.y, axis=1, inplace=True)

        X_train = df_train.iloc[:-(steps + term)]

        X_test = df_train.iloc[-(steps + term):]
        y_train = pd.DataFrame(Y, columns=[f'y+{i + 1}' for i in range(steps)])

        return X_train, y_train, X_test

    def data_split(df_X, df_y, rate: float = 0.7, recent: bool = False):

        '''
        데이터 스플릿 함수

        Parameters:
        df_X (dataframe): X 
        df_y (dataframe): y 
        rate (float) : ratio to split  ( default 0.7)
        recent (bool): whether to use the newest data as train data or not 

        Returns:
        X_train, X_val, y_train, y_val : train data has been splited into train and validation 
        '''

        if recent in [False, True]:
            pass
        else:
            return print('recent 파라미터에 잘못된 값이 들어갔습니다.[False, True] 범위에서 설정해주세요.')

        if recent == False:

            X_train, X_val, y_train, y_val = df_X.iloc[:int(len(df_X) * rate)], df_X.iloc[
                                                                                int(len(df_X) * rate):], df_y.iloc[:int(
                len(df_y) * rate)], df_y.iloc[int(len(df_y) * rate):]

        else:
            X_train, X_val, y_train, y_val = df_X.iloc[int(len(df_X) * (1 - rate)):], df_X.iloc[:int(
                len(df_X) * (1 - rate))], df_y.iloc[int(len(df_y) * (1 - rate)):], df_y.iloc[
                                                                                   :int(len(df_y) * (1 - rate))]

        return X_train, X_val, y_train, y_val

# cross val
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits, train_length, test_length, random_state=None):
        self.n_splits = n_splits
        self.train_length = train_length
        self.test_length = test_length
        self.random_state = random_state

    def get_n_splits(self, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            train_start = i * k_fold_size
            train_stop = train_start + self.train_length
            test_start = train_stop
            test_stop = test_start + self.test_length

            yield indices[train_start: train_stop], indices[test_start: test_stop]

# class and function definition 
class business_Metrics:
    def __init__(self, high_order, low_order):
        self.high_order = high_order
        self.low_order = low_order

    def custom_loss(self, y_true, y_pred):
        tensor_data = y_true - y_pred

        condition_negative = tensor_data < self.low_order
        condition_temp = (self.low_order >= tensor_data) & (tensor_data <= self.high_order)
        condition_positive = tensor_data >= self.high_order

        modified_values = tf.where(condition_positive, tf.abs(tensor_data), tensor_data - self.low_order)
        modified_values = tf.where(condition_temp, tf.constant(0, dtype=tf.float32), modified_values)
        modified_values = tf.where(condition_negative, tf.abs(tensor_data), tensor_data - self.high_order)

        mse = tf.reduce_mean(tf.square(modified_values)) # calculate adjusted MSE
        return mse

    def ml_loss(self, y_true, y_pred):

        ml_data =  np.array(list(pd.DataFrame(y_pred, columns=['y'])['y'])) - np.array(list(pd.DataFrame(y_true, columns=['y'])['y']))

        for idx, i in enumerate(ml_data):

            if i < self.low_order:
                ml_data[idx] = i - self.low_order

            elif i > self.high_order:
                ml_data[idx] = i - self.high_order

            else:
                ml_data[idx] = 0


        mse = np.mean((ml_data) ** 2)
        return np.sqrt(mse)

    def ml_loss_mape(self, y_true, y_pred):

        ml_data = np.array(list(pd.DataFrame(y_pred, columns=['y'])['y'])) - np.array(
            list(pd.DataFrame(y_true, columns=['y'])['y']))

        for idx, i in enumerate(ml_data):

            if self.high_order >= i >= self.low_order:
                y_pred[idx] = np.array(list(pd.DataFrame(y_true, columns=['y'])['y']))[idx]
            elif self.high_order < i:
                y_pred[idx] = y_pred[idx] - self.high_order

            else:
                y_pred[idx] = y_pred[idx] - self.low_order

        return mean_absolute_percentage_error(y_true, y_pred)

    def mse(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def shortage(self, y_true, y_pred):
        ml_data = np.array(list(pd.DataFrame(y_pred, columns=['y'])['y'])) - np.array(
            list(pd.DataFrame(y_true, columns=['y'])['y']))

        for idx, i in enumerate(ml_data):

            if i > 0:
                ml_data[idx] = 0


        mse = np.sum(ml_data)
        return mse



# Loss function with correlation coefficient that applies volatility 
class basic_model:
    def __init__(self,X_train, y_train, X_val, y_val, X_test=None, y_test=None, real_data=None ):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.real_data = real_data

    # reverse log 
    def reverse_ln_transformation(self, data):
        return np.exp(data)

    def sort_fold_train(self, model, n_splits=5, train_length=39,test_length=13):

        r_btscv = BlockingTimeSeriesSplit(n_splits=n_splits, train_length=train_length, test_length=test_length)
        # X_train['Order'] data to Numpy
        data = self.real_data.to_numpy()
        sorted_data = np.sort(data)

        a = abs(np.percentile(sorted_data, 50))  # median
        bm = business_Metrics(0,0)
        result_mse = []
        i = n_splits
        for train_index, val_index in r_btscv.split(self.X_train):
            try:
                X_train_fold, X_val_fold = self.X_train.values[train_index], self.X_train.values[val_index]
                y_train_fold, y_val_fold = self.y_train.values[train_index], self.y_train.values[val_index]
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)

                rmse = bm.ml_loss(y_val_fold, y_pred) * i/sum(range(1, 1+n_splits))
                result_mse.append(rmse)
                i -= 1
            except:
                pass

        return result_mse

    def accumulate_fold_train(self, model, n_splits=5):

        r_btscv = TimeSeriesSplit(n_splits=n_splits)
        data = self.real_data.to_numpy()
        sorted_data = np.sort(data)

        a = abs(np.percentile(sorted_data, 50))  # median
        bm = business_Metrics(0,0)
        result_mse = []
        i = n_splits
        for train_index, val_index in r_btscv.split(self.X_train):
            try:
                X_train_fold, X_val_fold = self.X_train.values[train_index], self.X_train.values[val_index]
                y_train_fold, y_val_fold = self.y_train.values[train_index], self.y_train.values[val_index]
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)

                rmse = bm.ml_loss(y_val_fold, y_pred) * i/sum(range(1, 1+n_splits))
                result_mse.append(rmse)
                i -= 1
            except:
                pass

        return result_mse

    def objective_lgbm_sort(self, trial):

        '''
        LGBM optuna 함수
        '''
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            "n_estimators": trial.suggest_int("n_estimators", 6000, 9000),
            'num_leaves': trial.suggest_int('num_leaves', 10,50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.15),
            'max_depth': trial.suggest_int('max_depth', 10, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 50),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 3),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 3)
        }

        model = LGBMRegressor(**params)
        result_mse = self.sort_fold_train(model)

        return sum(result_mse) / len(result_mse)

    def objective_rf_sort(self, trial):

        '''
        RF optuna 
        '''

        n_estimators = trial.suggest_int('n_estimators', 50, 100)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 20)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      random_state=42)
        model.fit(self.X_train, self.y_train)
        result_mse = self.sort_fold_train(model)

        return sum(result_mse) / len(result_mse)

    def objective_knn_sort(self, trial):

        '''
        KNN optuna 
        '''
        n_neighbors = trial.suggest_int('n_neighbors', 2, 30)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        result_mse = self.sort_fold_train(model)

        return sum(result_mse) / len(result_mse)

    def objective_lasso_sort(self, trial):

        '''
        LASSO optuna 
        '''
        alpha = trial.suggest_loguniform('alpha', 0.01, 3)
        model = Lasso(alpha=alpha)
        result_mse = self.sort_fold_train(model)
        return sum(result_mse) / len(result_mse)

    def objective_elasticnet_sort(self, trial):

        '''
        Elasticnet optuna 
        '''
        alpha = trial.suggest_loguniform('alpha', 0.01, 3)
        l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        result_mse = self.sort_fold_train(model)
        return sum(result_mse) / len(result_mse)

    def objective_ridge_sort(self, trial):
        '''
        Ridge optuna 
        '''

        alpha = trial.suggest_loguniform('alpha', 0.01, 3)
        model = Ridge(alpha=alpha)
        result_mse = self.sort_fold_train(model)
        return sum(result_mse) / len(result_mse)

    def objective_xgb_sort(self, trial):
        params = {
            'objective': 'reg:squarederror',  # Regression task
            'eval_metric': 'rmse',  # RMSE as the evaluation metric
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),

            'eta': trial.suggest_loguniform('eta', 0.01, 0.3),


        }

        model = xgb.XGBRegressor(**params)

        result_mse = self.sort_fold_train(model)

        return sum(result_mse) / len(result_mse)

    def objective_lgbm_accumulate(self, trial):

        '''
        LGBM optuna 
        '''
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',

            'num_leaves': trial.suggest_int('num_leaves', 10,50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.15),
            'max_depth': trial.suggest_int('max_depth', 10, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 50),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 3),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 3)
        }

        model = LGBMRegressor(**params)
        result_mse = self.accumulate_fold_train(model)

        return sum(result_mse) / len(result_mse)

    def objective_rf_accumulate(self, trial):

        '''
        RF optuna 
        '''


        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 20)
        model = RandomForestRegressor( max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      random_state=42)
        model.fit(self.X_train, self.y_train)
        result_mse = self.accumulate_fold_train(model)

        return sum(result_mse) / len(result_mse)

    def objective_knn_accumulate(self, trial):

        '''
        KNN optuna 
        '''
        n_neighbors = trial.suggest_int('n_neighbors', 2, 30)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        result_mse = self.accumulate_fold_train(model)

        return sum(result_mse) / len(result_mse)

    def objective_lasso_accumulate(self, trial):

        '''
        LASSO optuna 
        '''
        alpha = trial.suggest_loguniform('alpha', 0.01, 3)
        model = Lasso(alpha=alpha)
        result_mse = self.accumulate_fold_train(model)
        return sum(result_mse) / len(result_mse)

    def objective_elasticnet_accumulate(self, trial):

        '''
        Elasticnet optuna 
        '''
        alpha = trial.suggest_loguniform('alpha', 0.01, 3)
        l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        result_mse = self.accumulate_fold_train(model)
        return sum(result_mse) / len(result_mse)

    def objective_ridge_accumulate(self, trial):
        '''
        Ridge optuna 
        '''

        alpha = trial.suggest_loguniform('alpha', 0.01, 3)
        model = Ridge(alpha=alpha)
        result_mse = self.accumulate_fold_train(model)
        return sum(result_mse) / len(result_mse)

    def objective_xgb_accumulate(self, trial):
        params = {
            'objective': 'reg:squarederror',  # Regression task
            'eval_metric': 'rmse',  # RMSE as the evaluation metric
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
            'eta': trial.suggest_loguniform('eta', 0.01, 0.3),
        }

        model = xgb.XGBRegressor(**params)

        result_mse = self.accumulate_fold_train(model)

        return sum(result_mse) / len(result_mse)

    # Shapley value calculation function 
    def calculate_shapley_values(self, model, X_train):
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)
        mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
        shap.plots.bar(shap_values)
        return mean_abs_shap_values

    # function calculation variable importance ranking 
    def calculate_feature_importance(self, X_train, mean_abs_shap_values):
        feature_importance = list(zip(X_train.columns, mean_abs_shap_values))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_9_feature_importance = [i[0] for i in feature_importance[:9]]
        return top_9_feature_importance

    def study_model(self, objective, model, X_train, y_train, X_val, y_val):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)

        best_params = study.best_params
        best_model = model(**best_params)

        return best_model


        return y_train_pred, mape_train, y_val_pred, mape_val, best_model

    def last_train(self, best_model, a):
        best_model.fit(self.X_train,self.y_train)

        # predict (train)
        y_train = self.reverse_ln_transformation(self.y_train)
        y_val = self.reverse_ln_transformation(self.y_val)

        y_train_pred = self.reverse_ln_transformation(best_model.predict(self.X_train))

        bm = business_Metrics(0,0)
        mape_train = bm.ml_loss_mape(y_train, y_train_pred)

        # predit (val)
        y_val_pred = self.reverse_ln_transformation(best_model.predict(self.X_val))
        mape_val = bm.ml_loss_mape(y_val, y_val_pred)


        # graph  (train)
        plt.plot(y_train.index, y_train_pred, label='pred')
        plt.plot(y_train.index, y_train, label='act')

        plt.title('last_train_pred')
        plt.legend()

        plt.xlabel('time')
        plt.ylabel('Order')

        plt.show()

        plt.plot(y_val.index, y_val_pred, label='pred')
        plt.plot(y_val.index, y_val, label='act')

        plt.title('last_val_pred')
        plt.legend()

        plt.xlabel('time')
        plt.ylabel('Order')

        plt.show()

        y_val_pred = np.array(list(pd.DataFrame(y_val_pred, columns=['y'])['y']))
        y_train_pred = np.array(list(pd.DataFrame(y_train_pred, columns=['y'])['y']))

        return y_train_pred, mape_train, y_val_pred, mape_val


    def ma(self, period: int = 12, graph: bool = False):
        '''
        Prediction model using moving average

        Parameters:
        period (int): Period for calculating moving average (default 7)
        graph (bool): print graph (default False)

        Returns:
        y_pred (series): predict 
        '''
        y = pd.concat([self.y_train, self.y_val, self.y_test], axis=0)
        y_pred = y['y'].rolling(window=period).mean()
        y_pred = self.reverse_ln_transformation(y_pred)
        y = self.reverse_ln_transformation(y)
        y_high = self.reverse_ln_transformation(y + np.array(0.4))

        X_train = self.reverse_ln_transformation(self.X_train)
        # X_train['Order'] to Numpy
        data = self.X_train['Order'].to_numpy()

        sorted_data = np.sort(data)

        a = abs(np.percentile(sorted_data, 50))  # median

        bm = business_Metrics(0,0)

        mape = bm.ml_loss_mape(y.iloc[-len(self.y_test):], y_pred.iloc[-len(self.y_test):])
        mape_real = mean_absolute_percentage_error(y.iloc[-len(self.y_test):], y_pred.iloc[-len(self.y_test):])

        if graph == True:
            plt.plot(y_pred.iloc[-len(self.y_test):].index, y_pred.iloc[-len(self.y_test):], label='pred',  color='red')
            plt.plot(y_pred.iloc[-len(self.y_test):].index, y.iloc[-len(self.y_test):], label='act',  color='green')
            plt.plot(y_pred.iloc[-len(self.y_test):].index, y_high.iloc[-len(self.y_test):] , linestyle='--', color='green')

            plt.title('ma_pred')
            plt.legend()

            plt.xlabel('time')
            plt.ylabel('Order')

            plt.show()

        else:
            pass

        return np.array(list(y_pred.iloc[-len(self.y_test):])), mape, mape_real

    def lgbm(self, log: bool = False, cross_val = ['sort','accumulate']):

        if log == False:

            study_lgbm = optuna.create_study(direction="minimize")
            study_lgbm.optimize(self.objective_lgbm, n_trials=50)

            best_params = study_lgbm.best_params
            best_model = LGBMRegressor(**best_params)
            best_model.fit(pd.concat([self.X_train, self.X_val], axis=0), pd.concat([self.y_train, self.y_val], axis=0))
            y_train_pred = best_model.predict(self.X_train)
            y_val_pred = best_model.predict(self.X_val)
            y_test_pred = best_model.predict(self.X_test)
            mape_train = mean_absolute_percentage_error(self.y_train, y_train_pred)
            mape_val = mean_absolute_percentage_error(self.y_val, y_val_pred)
            mape_test = mean_absolute_percentage_error(self.y_test, y_test_pred)

            explainer = shap.Explainer(best_model)
            shap_values = explainer(self.X_train)
            shap.plots.bar(shap_values)

        else:
            if cross_val == 'sort':
                best_model = self.study_model(self.objective_lgbm_sort, LGBMRegressor, self.X_train, self.y_train, self.X_val, self.y_val)

            elif cross_val == 'accumulate':
                best_model = self.study_model(self.objective_lgbm_accumulate, LGBMRegressor, self.X_train, self.y_train,
                                              self.X_val, self.y_val)

            else:
                return 'cross_val 파라미터 값이 옳바르지 않습니다.'
            # X_train['Order'] to Numpy 
            data = self.real_data.to_numpy()

            sorted_data = np.sort(data)

            a = abs(np.percentile(sorted_data, 50))  
            y_train_pred, mape_train, y_val_pred, mape_val = self.last_train(best_model,a)
            # Calculate the average of the Shapley values ​​to obtain variable importance rankings.
            mean_abs_shap_values = self.calculate_shapley_values(best_model, self.X_train)

            # List the top 9 variable importance.
            top_9_feature_importance = self.calculate_feature_importance(self.X_train, mean_abs_shap_values)
            self.y_val = self.reverse_ln_transformation(self.y_val)
            self.y_train = self.reverse_ln_transformation(self.y_train)


        return y_train_pred, mape_train, y_val_pred, mape_val, top_9_feature_importance, best_model

    def rf(self, log: bool = False, cross_val = ['sort','accumulate']):

        if log == False:

            study_rf = optuna.create_study(direction="minimize")
            study_rf.optimize(self.objective_rf, n_trials=50)

            best_rf_params = study_rf.best_params
            best_rf_model = RandomForestRegressor(**best_rf_params)
            best_rf_model.fit(self.X_train, self.y_train)
            y_train_pred = best_rf_model.predict(self.X_train)
            y_val_pred = best_rf_model.predict(self.X_val)
            mape_train = mean_absolute_percentage_error(self.y_train, y_train_pred)
            mape_val = mean_absolute_percentage_error(self.y_val, y_val_pred)

        else:
            if cross_val == 'sort':
                best_model = self.study_model(self.objective_rf_sort, RandomForestRegressor, self.X_train, self.y_train,
                                              self.X_val, self.y_val)

            elif cross_val == 'accumulate':
                best_model = self.study_model(self.objective_rf_accumulate, RandomForestRegressor, self.X_train, self.y_train,
                                              self.X_val, self.y_val)

            else:
                return 'cross_val 파라미터 값이 옳바르지 않습니다.'
            # X_train['Order'] to NumPy
            data = self.real_data.to_numpy()
            sorted_data = np.sort(data)

            a = abs(np.percentile(sorted_data, 50))  # medain 
            y_train_pred, mape_train, y_val_pred, mape_val = self.last_train(best_model,a)
            # Calculate the average of the Shapley values ​​to obtain variable importance rankings.
            mean_abs_shap_values = self.calculate_shapley_values(best_model, self.X_train)

            # List the top 9 variable importance.
            top_9_feature_importance = self.calculate_feature_importance(self.X_train, mean_abs_shap_values)
            self.y_val = self.reverse_ln_transformation(self.y_val)
            self.y_train = self.reverse_ln_transformation(self.y_train)

        return y_train_pred, mape_train, y_val_pred, mape_val, top_9_feature_importance, best_model

    def knn(self, graph: bool = False, log: bool = False, cross_val = ['sort','accumulate']):

        if log == False:

            study_knn = optuna.create_study(direction="minimize")
            study_knn.optimize(self.objective_knn, n_trials=100)

            best_params = study_knn.best_params
            best_model = KNeighborsRegressor(**best_params)
            best_model.fit(self.X_train, self.y_train)
            y_train_pred = best_model.predict(self.X_train)
            y_val_pred = best_model.predict(self.X_val)
            mape_train = mean_absolute_percentage_error(self.y_train, y_train_pred)
            mape_val = mean_absolute_percentage_error(self.y_val, y_val_pred)


        else:
            study_knn = optuna.create_study(direction="minimize")
            study_knn.optimize(self.objective_knn, n_trials=100)

            best_params = study_knn.best_params
            best_model = KNeighborsRegressor(**best_params)
            best_model.fit(self.X_train, self.y_train)

            # predict(train)
            self.y_train = self.reverse_ln_transformation(self.y_train)
            self.y_val = self.reverse_ln_transformation(self.y_val)
            y_train_pred = self.reverse_ln_transformation(best_model.predict(self.X_train))
            mape_train = mean_absolute_percentage_error(self.y_train, y_train_pred)

            # predict  (val)
            y_val_pred = self.reverse_ln_transformation(best_model.predict(self.X_val))
            mape_val = mean_absolute_percentage_error(self.y_val, y_val_pred)

        if graph == True:
            # graph (train)
            plt.plot(self.y_train.index, y_train_pred, label='pred')
            plt.plot(self.y_train.index, self.y_train, label='act')

            plt.title('lgbm_train_pred')
            plt.legend()

            plt.xlabel('time')
            plt.ylabel('Order')

            plt.show()

            # graph (val)
            plt.plot(self.y_val.index, y_val_pred, label='pred')
            plt.plot(self.y_val.index, self.y_val, label='act')

            plt.title('lgbm_val_pred')
            plt.legend()

            plt.xlabel('time')
            plt.ylabel('Order')
            plt.show()

        else:
            pass

        return y_train_pred, mape_train, y_val_pred, mape_val

    def lasso(self, log: bool = False, cross_val = ['sort','accumulate']):

        if log == False:

            study_lasso = optuna.create_study(direction="minimize")
            study_lasso.optimize(self.objective_lasso, n_trials=50)

            best_params = study_lasso.best_params
            best_model = Lasso(**best_params)
            best_model.fit(self.X_train, self.y_train)
            y_train_pred = best_model.predict(self.X_train)
            y_val_pred = best_model.predict(self.X_val)
            mape_train = mean_absolute_percentage_error(self.y_train, y_train_pred)
            mape_val = mean_absolute_percentage_error(self.y_val, y_val_pred)


        else:
            if cross_val == 'sort':
                best_model = self.study_model(self.objective_lasso_sort, Lasso, self.X_train, self.y_train,
                                              self.X_val, self.y_val)

            elif cross_val == 'accumulate':
                best_model = self.study_model(self.objective_lasso_accumulate, Lasso, self.X_train,
                                              self.y_train,
                                              self.X_val, self.y_val)

            else:
                return 'cross_val 파라미터 값이 옳바르지 않습니다.'

            # X_train['Order'] to NumPy 
            data = self.real_data.to_numpy()

            sorted_data = np.sort(data)

            a = abs(np.percentile(sorted_data, 50))  # median
            y_train_pred, mape_train, y_val_pred, mape_val = self.last_train(best_model,a)
            # Choose a random test sample
            sample_idx = np.random.randint(0, len(self.X_val))
            sample = self.X_val.iloc[sample_idx]
            sample = pd.DataFrame([sample])

            # Initialize SHAP explainer for Ridge model
            explainer = shap.Explainer(best_model.predict, pd.concat([self.X_train, self.X_val], axis=0))

            # Calculate SHAP values for Ridge model
            shap_values = explainer(sample)

            # Visualize SHAP values for Ridge model
            shap.plots.bar(shap_values[0])

            mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)

            # List the top 9 variable importance.
            top_9_feature_importance = self.calculate_feature_importance(self.X_train, mean_abs_shap_values)
            self.y_val = self.reverse_ln_transformation(self.y_val)
            self.y_train = self.reverse_ln_transformation(self.y_train)


            return y_train_pred, mape_train, y_val_pred, mape_val, top_9_feature_importance, best_model


    def elasticnet(self, log: bool = False, cross_val = ['sort','accumulate']):

        if log == False:

            study_elasticnet = optuna.create_study(direction="minimize")
            study_elasticnet.optimize(self.objective_elasticnet, n_trials=50)

            best_params = study_elasticnet.best_params
            best_model = ElasticNet(**best_params)
            best_model.fit(self.X_train, self.y_train)
            y_train_pred = best_model.predict(self.X_train)
            y_val_pred = best_model.predict(self.X_val)
            mape_train = mean_absolute_percentage_error(self.y_train, y_train_pred)
            mape_val = mean_absolute_percentage_error(self.y_val, y_val_pred)


        else:
            if cross_val == 'sort':
                best_model = self.study_model(self.objective_elasticnet_sort, ElasticNet, self.X_train, self.y_train,
                                              self.X_val, self.y_val)

            elif cross_val == 'accumulate':
                best_model = self.study_model(self.objective_elasticnet_accumulate, ElasticNet, self.X_train,
                                              self.y_train,
                                              self.X_val, self.y_val)

            else:
                return 'cross_val 파라미터 값이 옳바르지 않습니다.'

            # X_train['Order'] to NumPy 
            data = self.real_data.to_numpy()

            sorted_data = np.sort(data)

            a = abs(np.percentile(sorted_data, 50))  # median
            y_train_pred, mape_train, y_val_pred, mape_val = self.last_train(best_model,a)

            # Choose a random test sample
            sample_idx = np.random.randint(0, len(self.X_val))
            sample = self.X_val.iloc[sample_idx]
            sample = pd.DataFrame([sample])

            # Initialize SHAP explainer for Ridge model
            explainer = shap.Explainer(best_model.predict, pd.concat([self.X_train, self.X_val], axis=0))

            # Calculate SHAP values for Ridge model
            shap_values = explainer(sample)

            # Visualize SHAP values for Ridge model
            shap.plots.bar(shap_values[0])

            mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)

            # List the top 9 variable importance.
            top_9_feature_importance = self.calculate_feature_importance(self.X_train, mean_abs_shap_values)
            self.y_val = self.reverse_ln_transformation(self.y_val)
            self.y_train = self.reverse_ln_transformation(self.y_train)


            return y_train_pred, mape_train, y_val_pred, mape_val, top_9_feature_importance, best_model

    def ridge(self, log: bool = False, cross_val = ['sort','accumulate']):

        if log == False:

            study_ridge = optuna.create_study(direction="minimize")
            study_ridge.optimize(self.objective_ridge, n_trials=50)

            best_params = study_ridge.best_params
            best_model = Ridge(**best_params)
            best_model.fit(self.X_train, self.y_train)
            y_train_pred = best_model.predict(self.X_train)
            y_val_pred = best_model.predict(self.X_val)
            mape_train = mean_absolute_percentage_error(self.y_train, y_train_pred)
            mape_val = mean_absolute_percentage_error(self.y_val, y_val_pred)


        else:
            if cross_val == 'sort':
                best_model = self.study_model(self.objective_ridge_sort, Ridge, self.X_train, self.y_train,
                                              self.X_val, self.y_val)

            elif cross_val == 'accumulate':
                best_model = self.study_model(self.objective_ridge_accumulate, Ridge, self.X_train,
                                              self.y_train,
                                              self.X_val, self.y_val)

            else:
                return 'cross_val 파라미터 값이 옳바르지 않습니다.'

            # X_train['Order'] to NumPy 
            data = self.real_data.to_numpy()

            sorted_data = np.sort(data)

            a = abs(np.percentile(sorted_data, 50))  # median
            y_train_pred, mape_train, y_val_pred, mape_val = self.last_train(best_model,a)
        # Choose a random test sample
        sample_idx = np.random.randint(0, len(self.X_val))
        sample = self.X_val.iloc[sample_idx]
        sample = pd.DataFrame([sample])

        # Initialize SHAP explainer for Ridge model
        explainer = shap.Explainer(best_model.predict, pd.concat([self.X_train, self.X_val], axis=0))

        # Calculate SHAP values for Ridge model
        shap_values = explainer(sample)

        # Visualize SHAP values for Ridge model
        shap.plots.bar(shap_values[0])

        mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)

        # List the top 9 variable importance.
        top_9_feature_importance = self.calculate_feature_importance(self.X_train, mean_abs_shap_values)
        self.y_val = self.reverse_ln_transformation(self.y_val)
        self.y_train = self.reverse_ln_transformation(self.y_train)


        return y_train_pred, mape_train, y_val_pred, mape_val, top_9_feature_importance, best_model

    def xgb(self, log: bool = False, cross_val = ['sort','accumulate']):

        if log == False:


            study_xgb = optuna.create_study(direction="minimize")
            study_xgb.optimize(self.objective_xgb, n_trials=100)

            best_params = study_xgb.best_params
            best_model = xgb.train(best_params, xgb.DMatrix(self.X_train, label=self.y_train))

            y_train_pred = best_model.predict(xgb.DMatrix(self.X_train))
            y_val_pred = best_model.predict(xgb.DMatrix(self.X_val))

            mape_train = mean_absolute_percentage_error(self.y_train, y_train_pred)
            mape_val = mean_absolute_percentage_error(self.y_val, y_val_pred)

        else:
            if cross_val == 'sort':
                best_model = self.study_model(self.objective_xgb_sort, xgb.XGBRegressor, self.X_train, self.y_train,
                                              self.X_val, self.y_val)

            elif cross_val == 'accumulate':
                best_model = self.study_model(self.objective_xgb_accumulate, xgb.XGBRegressor, self.X_train,
                                              self.y_train,
                                              self.X_val, self.y_val)

            else:
                return 'cross_val 파라미터 값이 옳바르지 않습니다.'
            # X_train['Order'] to NumPy 
            data = self.real_data.to_numpy()
            sorted_data = np.sort(data)

            a = abs(np.percentile(sorted_data, 50))  # median
            y_train_pred, mape_train, y_val_pred, mape_val = self.last_train(best_model, a)

            feature_importance = best_model.feature_importances_
            feature_names = self.X_train.columns

            # index of top 9 variables 
            top_9_indices = np.argsort(feature_importance)[::-1][:9]

            # column of top 9 variables 
            top_9_feature_names = feature_names[top_9_indices]

            # visualize the importance of variable 
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, feature_importance)
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            plt.title('Variable Importance')
            plt.show()


        return y_train_pred, mape_train, y_val_pred, mape_val, top_9_feature_names, best_model

    def pycaret(self, graph: bool = False):
        for name in self.y_train.columns.tolist():

            train = pd.concat([self.X_train, pd.DataFrame(self.y_train[name])], axis=1)
            val = pd.concat([self.X_val, pd.DataFrame(self.y_val[name])], axis=1)

            exp = setup(train, target=name, session_id=42, n_jobs=-1, fold_shuffle=False)

            # compare model and create 
            best_model = compare_models()

            # model tuning 
            tuned_model = tune_model(best_model)

            # model evaluation 
            evaluate_model(tuned_model)

            # model prediction 
            predictions = predict_model(tuned_model, data=self.X_val)

            y_pred = predictions['prediction_label']
            mape = mean_absolute_percentage_error(self.reverse_ln_transformation(self.y_val[name]),
                                                  self.reverse_ln_transformation(y_pred))

            if graph == True:
                plt.plot(self.y_val.loc[:, name].index, self.reverse_ln_transformation(y_pred), label='pred')
                plt.plot(self.y_val.loc[:, name].index, self.reverse_ln_transformation(self.y_val[name]), label='act')
               
                plt.title('pycaret_pred')
                plt.legend()

                plt.xlabel('time')
                plt.ylabel('Order')

                plt.show()

            else:
                pass

        return y_pred, mape

class ensemble:
    def __init__(self, model_pred, model_performance, model_name):
        '''
        :param model_pred: list
        :param model_performance: list
        :param model_name: dic
        '''

        self.model_pred = model_pred
        self.model_performance = model_performance
        self.model_name = model_name

    def weighted_average(self):
        model_performance
        sum(self.model_performance)

        return 0