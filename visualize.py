import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from datetime import datetime

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.constants import Time


def shape(path):
    df, labels = load_from_tsfile_to_dataframe(path, True)
    nlabels = len(labels)
    nrows = len(df) * Time.HOURS_IN_A_DAY
    ncols = len(df.columns)

    return nrows, ncols, nlabels


def graph_time_series(dim_name, units=""):
    df, _ = load_from_tsfile_to_dataframe(
        'data/BeijingPM25Quality_TRAIN.ts', True)
    df = df.rename(columns={"dim_0": "SO2", "dim_1": "NO2", "dim_2": "CO", "dim_3": "O3", "dim_4": "temperature",
                            "dim_5": "pressure", "dim_6": "dew_point", "dim_7": "rainfall", "dim_8": "windspeed"}, errors="raise")

    pd.set_option('display.max_columns', None)
    ym_list = []
    current_month = None
    dim_monthly = []
    dim_list = []

    for date in df[dim_name]:
        month = date.index[0].strftime('%Y-%m')
        if current_month == None:
            current_month = month
            ym_list.append(month)
        elif month != current_month:
            current_month = month
            dim = np.mean(dim_monthly)
            dim_list.append(dim)
            dim_monthly = []
            ym_list.append(month)

        dim_monthly.append(date.mean())

    dim_list.append(np.mean(dim_monthly))
    dim_series = pd.Series(data=dim_list, index=ym_list)

    ax = sns.lineplot(data=dim_series)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    plt.legend()
    plt.ylabel(dim_name + ' (' + units + ')')
    plt.xlabel('Year-Month')
    plt.title(dim_name + " Concentration")
    plt.savefig('graphs/' + dim_name + '.png')
    plt.close()


def graph_time_series(df, dim_name, units=""):
    df["YM"] = df["year"].astype(str) + "-" + df["month"].astype(str)

    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize': (14, 8)})

    ax = sns.lineplot(data=df, x='YM', y=dim_name,
                      hue='Station', palette='viridis',
                      legend='full', lw=3)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel(dim_name + " (" + units + ")")
    plt.xlabel('Year-Month')
    plt.title(dim_name + ' Z-scores')
    plt.savefig('graphs/normalized_' + dim_name + '_all_stations.png')
    plt.close()


def graph_all_time_series(path):
    df = pd.read_csv(path)
    df["YM"] = df["year"].astype(str) + "-" + df["month"].astype(str)
    df_monthly = df.groupby(['Station', 'YM', 'year', 'month']).mean()
    df_monthly = df_monthly[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                             'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].reset_index()

    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize': (14, 8)})

    ax = sns.lineplot(data=df_monthly, x='YM', y='PM2.5',
                      hue='Station', palette='viridis',
                      legend='full', lw=3)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel('PM2.5 (µg/m3)')
    plt.xlabel('Year-Month')
    plt.title('PM2.5 Concentration')
    plt.savefig('graphs/pm25_all_stations.png')
    plt.close()


# Requires: month and day are int
def find_day_mean_and_std(path, dim_name, month, day):
    df = pd.read_csv(path)
    df_daily_mean = df.groupby(['month', 'day']).mean(numeric_only=True)
    df_daily_mean = df_daily_mean[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                                   'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].reset_index()

    df_daily_std = df.groupby(['month', 'day']).std(numeric_only=True)
    df_daily_std = df_daily_std[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                                 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].reset_index()

    mean_data = df_daily_mean.loc[(df_daily_mean["month"] == month)
                                  & (df_daily_mean["day"] == day)]
    std_data = df_daily_std.loc[(df_daily_mean["month"] == month)
                                & (df_daily_mean["day"] == day)]
    mean = mean_data[dim_name].values[0]
    stdev = std_data[dim_name].values[0]

    return mean, stdev


def normalize(path):
    df = pd.read_csv(path)
    df_daily_mean = df.copy()
    df_daily_std = df.copy()
    for col in df_daily_mean.columns:
        if col not in ['No', 'year', 'month', 'day', 'hour', 'wd', 'Station']:
            df_daily_mean[col] = df.groupby(['month', 'day']).transform(
                'mean', numeric_only=True)[col]
            df_daily_std[col] = df.groupby(['month', 'day']).transform(
                'std', numeric_only=True)[col]

    df_normalized = df.copy()
    for col in df_daily_mean.columns:
        if col not in ['No', 'year', 'month', 'day', 'hour', 'wd', 'Station']:
            df_normalized[col] = (
                df[col] - df_daily_mean[col])/df_daily_std[col]

    df_normalized = df_normalized.fillna(0.0)

    return df_normalized


def normalize_ts(path):
    return None


def normalized_to_csv():
    SO2_list, NO2_list, CO_list, O3_list, temperature_list, pressure_list, dewpoint_list, rainfall_list, windspeed_list, PM_list = [
    ], [], [], [], [], [], [], [], [], []
    d_range = pd.date_range('2016-06-01',
                            '2017-02-28', freq='D')
    df_normalized = pd.read_csv('data/PRSA_Data_normalized_TEST.csv')
    for date in d_range:
        h_range = pd.date_range(date, periods=24, freq='H')
        rows = df_normalized.loc[(df_normalized['year'] == date.year) & (
            df_normalized['month'] == date.month) & (df_normalized['day'] == date.day)]
        SO2_list.append(pd.Series(rows['SO2'].values, index=h_range))
        NO2_list.append(pd.Series(rows['NO2'].values, index=h_range))
        CO_list.append(pd.Series(rows['CO'].values, index=h_range))
        O3_list.append(pd.Series(rows['O3'].values, index=h_range))
        temperature_list.append(pd.Series(rows['TEMP'].values, index=h_range))
        pressure_list.append(pd.Series(rows['PRES'].values, index=h_range))
        dewpoint_list.append(pd.Series(rows['DEWP'].values, index=h_range))
        rainfall_list.append(pd.Series(rows['RAIN'].values, index=h_range))
        windspeed_list.append(pd.Series(rows['WSPM'].values, index=h_range))
        PM_list.append(pd.Series(rows['PM2.5'].values, index=h_range))

    dict = {'SO2': SO2_list, 'NO2': NO2_list, 'CO': CO_list, 'O3': O3_list, 'TEMP': temperature_list,
            'PRES': pressure_list, 'DEWP': dewpoint_list, 'RAIN': rainfall_list, 'WSPM': windspeed_list, 'PM2.5': PM_list}
    df = pd.DataFrame(data=dict)

    pd.set_option('display.max_columns', None)
    print(df.head())
    df.to_csv('data/PRSA_Data_TEST2.csv', index=False)


def get_normalized_train():
    SO2_list, NO2_list, CO_list, O3_list, temperature_list, pressure_list, dewpoint_list, rainfall_list, windspeed_list, PM_list = [
    ], [], [], [], [], [], [], [], [], []
    d_range = pd.date_range('2013-03-01',
                            '2016-05-31', freq='D')
    df_normalized = pd.read_csv('data/PRSA_Data_normalized_TRAIN.csv')
    for date in d_range:
        h_range = pd.date_range(date, periods=24, freq='H')
        rows = df_normalized.loc[(df_normalized['year'] == date.year) & (
            df_normalized['month'] == date.month) & (df_normalized['day'] == date.day)]
        SO2_list.append(pd.Series(rows['SO2'].values, index=h_range))
        NO2_list.append(pd.Series(rows['NO2'].values, index=h_range))
        CO_list.append(pd.Series(rows['CO'].values, index=h_range))
        O3_list.append(pd.Series(rows['O3'].values, index=h_range))
        temperature_list.append(
            pd.Series(rows['TEMP'].values, index=h_range))
        pressure_list.append(pd.Series(rows['PRES'].values, index=h_range))
        dewpoint_list.append(pd.Series(rows['DEWP'].values, index=h_range))
        rainfall_list.append(pd.Series(rows['RAIN'].values, index=h_range))
        windspeed_list.append(
            pd.Series(rows['WSPM'].values, index=h_range))
        PM_list.append(pd.Series(rows['PM2.5'].values, index=h_range))

    dict = {'SO2': SO2_list, 'NO2': NO2_list, 'CO': CO_list, 'O3': O3_list, 'TEMP': temperature_list,
            'PRES': pressure_list, 'DEWP': dewpoint_list, 'RAIN': rainfall_list, 'WSPM': windspeed_list, 'PM2.5': PM_list}
    return pd.DataFrame(data=dict)


def get_normalized_test():
    SO2_list, NO2_list, CO_list, O3_list, temperature_list, pressure_list, dewpoint_list, rainfall_list, windspeed_list, PM_list = [
    ], [], [], [], [], [], [], [], [], []
    d_range = pd.date_range('2016-06-01',
                            '2017-02-28', freq='D')
    df_normalized = pd.read_csv('data/PRSA_Data_normalized_TEST.csv')
    for date in d_range:
        h_range = pd.date_range(date, periods=24, freq='H')
        rows = df_normalized.loc[(df_normalized['year'] == date.year) & (
            df_normalized['month'] == date.month) & (df_normalized['day'] == date.day)]
        SO2_list.append(pd.Series(rows['SO2'].values, index=h_range))
        NO2_list.append(pd.Series(rows['NO2'].values, index=h_range))
        CO_list.append(pd.Series(rows['CO'].values, index=h_range))
        O3_list.append(pd.Series(rows['O3'].values, index=h_range))
        temperature_list.append(
            pd.Series(rows['TEMP'].values, index=h_range))
        pressure_list.append(pd.Series(rows['PRES'].values, index=h_range))
        dewpoint_list.append(pd.Series(rows['DEWP'].values, index=h_range))
        rainfall_list.append(pd.Series(rows['RAIN'].values, index=h_range))
        windspeed_list.append(
            pd.Series(rows['WSPM'].values, index=h_range))
        PM_list.append(pd.Series(rows['PM2.5'].values, index=h_range))

    dict = {'SO2': SO2_list, 'NO2': NO2_list, 'CO': CO_list, 'O3': O3_list, 'TEMP': temperature_list,
            'PRES': pressure_list, 'DEWP': dewpoint_list, 'RAIN': rainfall_list, 'WSPM': windspeed_list, 'PM2.5': PM_list}
    return pd.DataFrame(data=dict)


if __name__ == "__main__":
    # df = pd.read_csv('data/PRSA_Data_merged.csv')
    # df = df.groupby(['year', 'month', 'day', 'hour'])
    print("hello")
