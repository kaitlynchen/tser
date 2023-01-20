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


if __name__ == "__main__":
    graph_time_series("CO", "µg/m3")
