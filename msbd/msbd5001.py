# coding=utf-8

import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

warnings.filterwarnings('ignore')


def read_data(DATA_DIR_PATH):
    """
    read data
    """
    df_train = pd.read_csv(DATA_DIR_PATH + 'train.csv', index_col=0)
    df_test = pd.read_csv(DATA_DIR_PATH + 'test.csv', index_col=0)
    return df_train, df_test


def fill_na(input_df):
    """
    fill for missing speed with mean value
    """
    miss_dates = [
        pd.to_datetime('2017-01-16 22'),
        pd.to_datetime('2017-01-16 23'),
        pd.to_datetime('2017-01-17 00'),
        pd.to_datetime('2017-01-17 01'),
        pd.to_datetime('2017-01-18 03'),
        pd.to_datetime('2017-01-18 04'),
        pd.to_datetime('2017-01-18 05'),
        pd.to_datetime('2017-01-18 06'),
        pd.to_datetime('2017-01-18 07'),
        pd.to_datetime('2017-07-15 16')
    ]

    append_values = []
    for date in miss_dates:
        y = date.year
        m = date.month
        d = date.day
        h = date.hour
        tmp_ = input_df.query('year == 2017 and hour == {}'.format(h))
        estimate_speed = tmp_['speed'].mean()
        append_values.append([date, estimate_speed, date.date(), y, m, d, h])

    output_df = input_df.append(pd.DataFrame(append_values, columns=total_df.columns))

    output_df = output_df.reset_index(drop=True).sort_values('datetime')
    return output_df


if __name__ == '__main__':
    train_df, test_df = read_data('./')

    # change columns date type
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])

    # merge train_df and test_df
    total_df = pd.concat((train_df, test_df))
    total_df.rename(columns={'date': 'datetime'}, inplace=True)
    total_df['date'] = total_df['datetime'].apply(lambda x: x.date())

    # generate time feature
    total_df['year'] = total_df['datetime'].apply(lambda x: x.year)
    total_df['month'] = total_df['datetime'].apply(lambda x: x.month)
    total_df['day'] = total_df['datetime'].apply(lambda x: x.day)
    total_df['hour'] = total_df['datetime'].apply(lambda x: x.hour)

    # fill missing
    total_df = fill_na(total_df)

    # split data from 2017 and use it as train dataset
    df_2017 = total_df[total_df['year'] == 2017]

    # add slice features and difference features
    for i in range(1, 30):
        df_2017['speed_{}_24'.format(i)] = df_2017['speed'].shift(i * 24)

    for i in range(1, 30):
        df_2017['speed_{}'.format(i)] = df_2017['speed'].shift(i)

    df_2017.dropna(inplace=True)

    df_2017 = df_2017.drop(columns=['date', 'year', 'month', 'day', 'hour'])
    target = df_2017['speed']
    features = df_2017.drop(columns=['datetime', 'speed'])

    gbd = GradientBoostingRegressor(random_state=2020)

    print("Start training...")

    # train
    gbd.fit(features, target)

    # predict
    total_df_cp = total_df.copy().drop(columns=['date', 'month', 'day', 'hour'])

    total_df_cp.set_index('datetime', inplace=True)

    length = total_df_cp.shape[0]

    print("\nStart predicting...\n")

    for i in tqdm(range(length)):
        # find nan speed then predict
        if total_df_cp.iloc[i]['year'] == 2018 and pd.isnull(total_df_cp.iloc[i]['speed']):
            tmp = total_df_cp.copy().iloc[i - 30 * 24: i + 1]

            for t in range(1, 30):
                tmp['speed_{}_24'.format(t)] = tmp['speed'].shift(t * 24)

            for t in range(1, 30):
                tmp['speed_{}'.format(t)] = tmp['speed'].shift(t)

            features = tmp.iloc[-1].drop(['speed', 'year'])

            pred = gbd.predict(features.values.reshape(1, -1))[0]

            total_df_cp.loc[total_df_cp.iloc[i].name, 'speed'] = pred

    speed = test_df.merge(
        total_df_cp.reset_index()[['datetime', 'speed']], left_on='date', right_on='datetime')[['speed']]

    # save data
    final = pd.read_csv('./test.csv', index_col=0)
    final['speed'] = speed

    final.to_csv('./test.csv')
