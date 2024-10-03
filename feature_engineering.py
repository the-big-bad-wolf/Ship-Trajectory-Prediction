import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

    # Calculate time difference within each vesselId group
    train_df['time_diff'] = train_df.groupby('vesselId')['time'].diff().dt.total_seconds()

    # Fill the first value of each group with 0 or another value if necessary
    train_df['time_diff'].fillna(0, inplace=True)

    # Add features for month, day, hour, minute and second
    train_df['month'] = train_df['time'].dt.month
    train_df['day'] = train_df['time'].dt.day
    train_df['hour'] = train_df['time'].dt.hour
    train_df['minute'] = train_df['time'].dt.minute
    train_df['second'] = train_df['time'].dt.second

    # Add previous longitude and latitude for each vesselId
    train_df['prev_longitude'] = train_df.groupby('vesselId')['longitude'].shift(1)
    train_df['prev_latitude'] = train_df.groupby('vesselId')['latitude'].shift(1)

    # Fill the first value of each group with the current value if necessary
    train_df['prev_longitude'].fillna(train_df['longitude'], inplace=True)
    train_df['prev_latitude'].fillna(train_df['latitude'], inplace=True)

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

    # Add features for month, day, hour, minute and second
    test_df_1['month'] = test_df_1['time'].dt.month
    test_df_1['day'] = test_df_1['time'].dt.day
    test_df_1['hour'] = test_df_1['time'].dt.hour
    test_df_1['minute'] = test_df_1['time'].dt.minute
    test_df_1['second'] = test_df_1['time'].dt.second

    # Group by vesselId and find the latest longitude and latitude based on the latest timestamp
    last_known_values = train_df.sort_values('time').groupby('vesselId').tail(1)

    test_df_1 = test_df_1.sort_values(['vesselId', 'time'])

    # Merge the last known longitude and latitude values with ais_test_df
    test_df_2 = pd.merge(test_df_1, last_known_values[['vesselId', 'longitude', 'latitude']], on='vesselId', how='left')

    # Initialize prev_longitude and prev_latitude as the last known values
    test_df_2['prev_longitude'] = test_df_2['longitude']
    test_df_2['prev_latitude'] = test_df_2['latitude']

    # Drop the temporary columns 'longitude_last' and 'latitude_last'
    test_df_2.drop(['longitude', 'latitude'], axis=1, inplace=True)

    return test_df_2

