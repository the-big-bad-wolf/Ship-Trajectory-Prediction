import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic
import pickle

def fill_train(df): 

    # fill missing values
    df["moored"] = df["navstat"].apply(
        lambda x: 1 if (x in [1, 5, 6]) else 0
    )

    df["restricted"] = df["navstat"].apply(
        lambda x: 1 if (x in [2, 3, 4, 7, 14]) else 0
    )

    df.loc[(df["sog"] == 102.3) & (df["moored"] == 1), "sog"] = 0

    df['sog'] = df.groupby('vesselId')['sog'].transform(
    lambda group: group.mask(group >= 102.3).interpolate(limit=3).ffill().bfill()
    )

    # Normalize the SOG feature
    df["sog"] = df["sog"] / 102.2

    # Interpolate the COG value between the previous and next value per vesselId if it is 360 or above
    df['cog'] = df.groupby('vesselId')['cog'].transform(
    lambda group: group.mask(group >= 360).interpolate(limit=3).ffill().bfill()
    )

    # Interpolate the heading value between the previous and next value per vesselId if it is 360 or above
    df['heading'] = df.groupby('vesselId')['heading'].transform(
    lambda group: group.mask(group >= 360).interpolate(limit=3).ffill().bfill()
    )

    # Normalize the heading and COG feature
    #train_df["cog"] = train_df["cog"] / 360
    df['cog_sin'] = np.sin(df['cog'] * np.pi / 180)
    df['cog_cos'] = np.cos(df['cog'] * np.pi / 180)

    #train_df["heading"] = train_df["heading"] / 360
    df['heading_sin'] = np.sin(df['heading'] * np.pi / 180)
    df['heading_cos'] = np.cos(df['heading'] * np.pi / 180)

    # legge til heading_sin og heading_cos?

    df["rot"] = (
    df.groupby("vesselId", as_index=False)  
    .apply(
        lambda group: group["rot"]  
        .mask(group["rot"].isin([-127, 127, 128]))  
        .interpolate(limit_area="inside", limit=1)  
        .fillna(0)  
    )
    .reset_index(drop=True)  
    )

    # Drop features that are not needed
    df.drop(['navstat', 'etaRaw', 'portId', 'heading', 'cog'], axis = 1, inplace = True)

    return df

def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Normalized distance
    normalized_distance = R*c  # c is already a fraction of the Earth's radius
    return normalized_distance

def engineer_train(pre_processed_df):
    # Convert the time column to datetime
    pre_processed_df['time'] = pd.to_datetime(pre_processed_df['time'])

    # Sort by vesselId and timestamp
    pre_processed_df.sort_values(by=['vesselId', 'time'], inplace=True)

    # Calculate time difference within each vesselId group
    pre_processed_df['time_diff'] = pre_processed_df.groupby('vesselId')['time'].diff().dt.total_seconds()

    # Add previous longitude and latitude for each vesselId
    pre_processed_df['prev_longitude'] = pre_processed_df.groupby('vesselId')['longitude'].shift(1)
    pre_processed_df['prev_latitude'] = pre_processed_df.groupby('vesselId')['latitude'].shift(1)

    # Add delta_longitude and delta_latitude which is the difference between the previous and previous previous longitude and latitude
    pre_processed_df['delta_longitude'] = pre_processed_df.groupby('vesselId')['prev_longitude'].diff()
    pre_processed_df['delta_latitude'] = pre_processed_df.groupby('vesselId')['prev_latitude'].diff()

    pre_processed_df['prev_latitude_lag'] = pre_processed_df.groupby('vesselId')['prev_latitude'].shift(1)
    pre_processed_df['prev_longitude_lag'] = pre_processed_df.groupby('vesselId')['prev_longitude'].shift(1)

    pre_processed_df['haversine_distance'] = pre_processed_df.apply(
    lambda row: haversine(
        row['prev_latitude'], 
        row['prev_longitude'],
        row['prev_latitude_lag'], 
        row['prev_longitude_lag']
    ) if pd.notnull(row['prev_latitude_lag']) and pd.notnull(row['prev_longitude_lag']) else np.nan,
    axis=1
    )

    # Add previous speed for each vesselId
    pre_processed_df['prev_sog'] = pre_processed_df.groupby('vesselId')['sog'].shift(1)

    # Add previous course for each vesselId
    pre_processed_df['prev_cog_cos'] = pre_processed_df.groupby('vesselId')['cog_cos'].shift(1)
    pre_processed_df['prev_cog_sin'] = pre_processed_df.groupby('vesselId')['cog_sin'].shift(1)

    # Add previous heading for each vesselId
    pre_processed_df['prev_heading_cos'] = pre_processed_df.groupby('vesselId')['heading_cos'].shift(1)
    pre_processed_df['prev_heading_sin'] = pre_processed_df.groupby('vesselId')['heading_sin'].shift(1)

    # Add moored or not feature if Navstat is 1, 5 (look into 9-13)
    pre_processed_df['prev_moored'] = pre_processed_df.groupby('vesselId')['moored'].shift(1)

    pre_processed_df.dropna(inplace=True)

    return pre_processed_df

def engineer_test(train_df, test_df):
    # Convert the time column to datetime
    test_df['time'] = pd.to_datetime(test_df['time'])

    # Group by vesselId and find the latest longitude and latitude based on the latest timestamp
    last_known_values = train_df.sort_values('time').groupby('vesselId').tail(1)

    last_known_values.rename(columns={'time': 'last_time'}, inplace=True)

    # Merge the last known values with the test set
    test_df = test_df.merge(last_known_values[['vesselId', 'last_time', 'longitude', 'latitude', 'sog', 'cog_cos', 'cog_sin', 'heading_cos', 'heading_sin', 'moored', 'prev_longitude', 'prev_latitude']], on='vesselId', how='left')

    # Calculate the time difference between the last known time and the current time
    test_df['time_diff'] = (test_df['time'] - test_df['last_time']).dt.total_seconds()

    # Add delta_longitude and delta_latitude which is the difference between the previous and previous previous longitude and latitude
    test_df['delta_longitude'] = test_df['longitude'] - test_df['prev_longitude']
    test_df['delta_latitude'] = test_df['latitude'] - test_df['prev_latitude']

    test_df['prev_latitude_lag'] = test_df['prev_latitude']
    test_df['prev_longitude_lag'] = test_df['prev_longitude']

    test_df.drop(['last_time', 'prev_longitude', 'prev_latitude'], axis=1, inplace=True)

    # Add previous longitude and latitude for each vesselId
    test_df['prev_longitude'] = test_df['longitude'].values
    test_df['prev_latitude'] = test_df['latitude'].values

    #display(test_df)

    test_df['haversine_distance'] = test_df.apply(lambda row: haversine(row['prev_latitude'], row['prev_longitude'], row['prev_latitude_lag'], row['prev_longitude_lag']) if pd.notnull(row['prev_latitude']) and pd.notnull(row['prev_longitude']) else np.nan,
    axis=1)

    # Add previous speed for each vesselId
    test_df['prev_sog'] = test_df['sog']

    # Add previous course for each vesselId
    test_df['prev_cog_cos'] = test_df['cog_cos']
    test_df['prev_cog_sin'] = test_df['cog_sin']

    # Add previous heading for each vesselId
    test_df['prev_heading_cos'] = test_df['heading_cos']
    test_df['prev_heading_sin'] = test_df['heading_sin']

    # Add moored or not feature if Navstat is 1, 5 (look into 9-13)
    test_df['prev_moored'] = test_df['moored'].values

    test_df.drop(['sog', 'cog_cos', 'cog_sin', 'heading_cos', 'heading_sin', 'moored', 'longitude', 'latitude'], axis=1, inplace=True)


    return test_df

def add_vessel_type(vessel_df, df):
    
    vessel_subset = vessel_df.loc[:, ['vesselId', 'length']]

    # Add the vessel type feature to the dataframe
    df_total = df.merge(vessel_subset, on='vesselId', how='left')

    # Add deep-sea boat feature. If length > 200, assign 1, else 0
    df_total['deep_sea'] = df_total['length'].apply(lambda x: 1 if x > 160 else 0)

    #df_total['length'] = df_total['length'] / 300

    # Drop the length column
    #df_total.drop('length', axis=1, inplace=True)

    return df_total

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


def process_data(train_csv, test_csv, vessel_csv):
    '''
    Process the raw data: pre process, feature engineering

    Args:
    training: str, path to the training data
    test: str, path to the test data

    '''
    # Load the data
    ais_train_df = pd.read_csv(train_csv, sep='|')
    ais_test_df = pd.read_csv(test_csv, sep=',')
    vessel_df = pd.read_csv(vessel_csv, sep='|')

    # Merge the vessel information
    ais_train_df = add_vessel_type(vessel_df, ais_train_df)
    ais_test_df = add_vessel_type(vessel_df, ais_test_df)

    # Pre process missing data in training set
    ais_train_df = fill_train(ais_train_df)

    # Feature engineering
    features_train_df = engineer_train(ais_train_df)
    features_test_df = engineer_test(features_train_df, ais_test_df)

    # Standardize the data
    #features_train_df = pp.standardize(features_train_df)

    return features_train_df, features_test_df

def calculate_distance(row):
    # Calculate the weighted distance for each row
    distance = geodesic((row['latitude'], row['longitude']), 
                        (row['latitude_predicted'], row['longitude_predicted'])).meters
    # Weight the distance by the scaling factor
    weighted_distance = distance * 0.3
    return weighted_distance

def calculate_score(solution_submission):
        # Calculate the weighted distance for each row
        solution_submission['weighted_distance'] = solution_submission.apply(calculate_distance, axis=1)

        weighted_distance = solution_submission['weighted_distance'].mean() / 1000.0

        return weighted_distance

def evaluate_model(y_pred, y_true):
    '''
    Evaluate the model using the validation set

    Args:
    y_pred: array, the predicted values
    y_true: array, the true values

    '''

    # Prepare output
    #y_cat_prep = val.prepare_output(y_pred, y_true)

    y_true['longitude_predicted'] = y_pred['longitude_predicted'].values
    y_true['latitude_predicted'] = y_pred['latitude_predicted'].values

    # Calculate score
    score = calculate_score(y_true)

    return score

def run_random_forest(model_lat, model_long, train_csv, test_csv, validation=True):
    model_features = ['time_diff', 'prev_longitude', 'prev_latitude', 'prev_longitude_lag', 'prev_latitude_lag', 'haversine_distance', 'delta_longitude', 'delta_latitude', 'prev_sog', 'prev_cog_cos', 'prev_cog_sin', 'prev_heading_cos', 'prev_heading_sin', 'prev_moored', 'deep_sea']
    test_features = ['ID', 'vesselId', 'scaling_factor', 'time_diff', 'prev_longitude', 'prev_latitude', 'delta_longitude', 'delta_latitude', 'prev_sog', 'prev_cog_cos', 'prev_cog_sin', 'prev_heading_cos', 'prev_heading_sin', 'prev_moored', 'deep_sea']

    #train_processed, test_processed = process_data(train_csv, test_csv, vessel_csv)
    train_processed = pd.read_csv('train_processed.csv')
    test_processed = pd.read_csv('test_processed.csv')

    #train_processed = train_processed[[model_features]]
    #test_processed = test_processed[[test_features]]


    if validation:
        subset_train, subset_val = split_train_validation(train_processed)
        model_long.fit(subset_train[model_features], subset_train['longitude'])
        model_lat.fit(subset_train[model_features], subset_train['latitude'])

        # Evaluate the model
        long_pred = model_long.predict(subset_val[model_features])
        lat_pred = model_lat.predict(subset_val[model_features])

        prediction = np.column_stack((long_pred, lat_pred))
        prediction_df = pd.DataFrame(prediction, columns=['longitude_predicted', 'latitude_predicted'])

        score = evaluate_model(prediction_df, subset_val)

        return score, prediction_df, model_long, model_lat


    
    else:
        model_long.fit(train_processed[model_features], train_processed['longitude'])
        model_lat.fit(train_processed[model_features], train_processed['latitude'])

        long_pred = model_long.predict(test_processed[model_features])
        lat_pred = model_lat.predict(test_processed[model_features])

        prediction = np.column_stack((test_processed['ID'].values, long_pred, lat_pred))
        prediction_df = pd.DataFrame(prediction, columns=['ID', 'longitude_predicted', 'latitude_predicted'])

        return None, prediction_df, model_long, model_lat


if __name__ == "__main__":
    rbr_score, rbr_predictions, rbr_long_model, rbr_lat_model = run_random_forest(RandomForestRegressor(n_estimators=100, random_state=42), 
                                                                                  RandomForestRegressor(n_estimators=100, random_state=42), 
                                                                                  'training_set.csv', 'test_set.csv', validation=False)
    

    # Save (dump) the model
    with open('rf_model.pkl', 'wb') as file:
        pickle.dump(rbr_long_model, file)
        pickle.dump(rbr_lat_model, file)
    
    with open('rf_predictions.pkl', 'wb') as file:
        pickle.dump(rbr_predictions, file)
        pickle.dump(rbr_score, file)