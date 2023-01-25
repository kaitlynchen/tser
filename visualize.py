import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from utils.data_loader import load_from_tsfile_to_dataframe


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


def graph_time_series(path, dim_name, units=""):
    df = pd.read_csv(path, na_values='NA')
    print(df.head())


def graph_all_time_series(path):
    df = pd.read_csv(path)
    df["YM"] = df["year"].astype(str) + "-" + df["month"].astype(str)
    df_monthly = df.groupby(['Station', 'YM', 'year', 'month']).mean()
    df_monthly = df_monthly[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                             'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].reset_index()

    print(df_monthly.head())

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


# Requires: date is in the form MM-DD (e.g. 1-1, 12-1, 5-30)
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


if __name__ == "__main__":
    # graph_time_series("CO", "µg/m3")
    # graph_all_time_series("data/PRSA_Data_merged.csv")
    mean, std = find_day_mean_and_std("data/PRSA_Data_merged.csv", "SO2", 1, 1)
    print("Mean: ", mean)
    print("Standard deviation: ", std)
