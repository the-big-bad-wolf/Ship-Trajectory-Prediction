import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pre_processing as pp

def transform_categorical(train_df, test_df, column):
    le = LabelEncoder()
    le.fit(train_df[column])
    train_df[column] = le.transform(train_df[column])
    test_df[column] = le.transform(test_df[column])
    return train_df, test_df


def feature_engineer_train(train_df):
    # Convert the time column to datetime
    train_df['time'] = pd.to_datetime(train_df['time'])

    # Sort by vesselId and timestamp
    train_df.sort_values(by=['vesselId', 'time'], inplace=True)

    # Preprocess the data
    train_df = pp.pre_process(train_df)

    # Calculate time difference within each vesselId group
    train_df['time_diff'] = train_df.groupby('vesselId')['time'].diff().dt.total_seconds()

    # Fill the first value of each group with 0 or another value if necessary
    train_df['time_diff'].fillna(0, inplace=True)

    # Add previous longitude and latitude for each vesselId
    train_df['prev_longitude'] = train_df.groupby('vesselId')['longitude'].shift(1)
    train_df['prev_latitude'] = train_df.groupby('vesselId')['latitude'].shift(1)

    # Fill the first value of each group with the current value if necessary
    train_df['prev_longitude'].fillna(train_df['longitude'], inplace=True)
    train_df['prev_latitude'].fillna(train_df['latitude'], inplace=True)

    # Calculate the distance between the previous and current location with geodesic distance
    #train_df['distance_travelled'] = train_df.apply(lambda x: geodesic((x['prev_latitude'], x['prev_longitude']), (x['latitude'], x['longitude'])).meters, axis=1)

    # Add previous speed for each vesselId
    train_df['prev_sog'] = train_df.groupby('vesselId')['sog'].shift(1)

    # Fill the first value of each group with the current value if necessary
    train_df['prev_sog'].fillna(train_df['sog'], inplace=True)

    # Add previous course for each vesselId
    train_df['prev_cog'] = train_df.groupby('vesselId')['cog'].shift(1)

    # Fill the first value of each group with the current value if necessary
    train_df['prev_cog'].fillna(train_df['cog'], inplace=True)

    # Add previous heading for each vesselId
    train_df['prev_heading'] = train_df.groupby('vesselId')['heading'].shift(1)

    # Fill the first value of each group with the current value if necessary
    train_df['prev_heading'].fillna(train_df['heading'], inplace=True)

    # Add moored or not feature if Navstat is 1, 5 (look into 9-13)
    train_df['prev_moored'] = train_df.groupby('vesselId')['moored'].shift(1)

    # Fill the first value of each group with the current value if necessary
    train_df['prev_moored'].fillna(train_df['moored'], inplace=True)

    return train_df

def split_train_validation(train_df):

    # Ensure that time column is datetime
    train_df['time'] = pd.to_datetime(train_df['time'])

    # Sort the data by time
    train_df = train_df.sort_values(by='time')

    # Identify the date for splitting 
    split_date = train_df['time'].max() - pd.Timedelta(days=5)

    # Train set: data before the split date
    train_df_split = train_df[train_df['time'] < split_date]

    # Test set: data within the last 5 days
    validation_df = train_df[train_df['time'] >= split_date]

    return train_df_split, validation_df

def choose_features(df, features, target, category='', categorical=False):
    X = df[features]
    y = df[target]

    if categorical:
        X[category] = X[category].astype('category')

    return X, y

def feature_engineer_test(train_df, test_df):
    # Convert the time column to datetime
    test_df['time'] = pd.to_datetime(test_df['time'])

    # Calculate the time difference between the last time in the training set and the time of each vessel in the test set
    # Get the latest time for each vesselId in the training set
    latest_time_train = train_df.groupby('vesselId')['time'].max().reset_index()
    latest_time_train.columns = ['vesselId', 'latest_time']  # Renaming for clarity

    # Merge the latest time into the test set based on vesselId
    test_df_1 = pd.merge(test_df, latest_time_train, on='vesselId', how='left')

    # Step 4: Calculate the time difference
    test_df_1['time_diff'] = (test_df_1['time'] - test_df_1['latest_time']).dt.total_seconds()

    # Calculate time difference within each vesselId group
    test_df['time_diff'] = test_df.groupby('vesselId')['time'].diff().dt.total_seconds()

    # Drop the temporary column 'latest_time'
    test_df_1.drop('latest_time', axis=1, inplace=True)

    # Group by vesselId and find the latest longitude and latitude based on the latest timestamp
    last_known_values = train_df.sort_values('time').groupby('vesselId').tail(1)

    test_df_1 = test_df_1.sort_values(['vesselId', 'time'])

    # Merge the last known longitude and latitude values with ais_test_df
    test_df_2 = pd.merge(test_df_1, last_known_values[['vesselId', 'longitude', 'latitude', 'sog', 'cog', 'heading', 'moored']], on='vesselId', how='left')

    # Initialize prev_longitude and prev_latitude as the last known values
    test_df_2['prev_longitude'] = test_df_2['longitude']
    test_df_2['prev_latitude'] = test_df_2['latitude']
    test_df_2['prev_sog'] = test_df_2['sog']
    test_df_2['prev_cog'] = test_df_2['cog']
    test_df_2['prev_heading'] = test_df_2['heading']
    test_df_2['prev_moored'] = test_df_2['moored']

    # Drop the temporary columns 'longitude_last' and 'latitude_last'
    test_df_2.drop(['longitude', 'latitude', 'sog', 'cog', 'heading', 'moored'], axis=1, inplace=True)

    return test_df_2

