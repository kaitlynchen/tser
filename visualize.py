import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from utils.data_loader import load_from_tsfile_to_dataframe


def main():
    df, _ = load_from_tsfile_to_dataframe(
        'data/BeijingPM25Quality_TRAIN.ts', True)
    df = df.rename(columns={"dim_0": "SO2", "dim_1": "NO2", "dim_2": "CO", "dim_3": "O3", "dim_4": "temperature",
                            "dim_5": "pressure", "dim_6": "dew_point", "dim_7": "rainfall", "dim_8": "windspeed"}, errors="raise")

    pd.set_option('display.max_columns', None)
    ym_list = []
    current_month = None
    so2_monthly = []
    so2_list = []

    for date in df["SO2"]:
        month = date.index[0].strftime('%Y-%m')
        if current_month == None:
            current_month = month
            ym_list.append(month)
        elif month != current_month:
            current_month = month
            so2 = np.mean(so2_monthly)
            so2_list.append(so2)
            so2_monthly = []
            ym_list.append(month)

        so2_monthly.append(date.mean())

    so2_list.append(np.mean(so2_monthly))
    print(len(ym_list))
    so2_series = pd.Series(data=so2_list, index=ym_list)

    # assert len(day_list) == len(so2_list)
    # df["Date"] = day_list
    # df["SO2"] = so2_list

    # sns.set_style('darkgrid')
    # sns.set(rc={'figure.figsize': (14, 8)})

    # ax = sns.lineplot(data=df, x='Date', y='SO2', palette='viridis',
    #                   legend='full', lw=3)
    ax = sns.lineplot(data=so2_series)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    # plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel('PM2.5 (µg/m3)')
    plt.xlabel('Year-Month')
    plt.show()


if __name__ == "__main__":
    main()
