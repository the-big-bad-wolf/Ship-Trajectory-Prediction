import pandas as pd


def pre_process(training_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processing of the data
    :param data: data to be pre-processed
    :return: pre-processed data
    """
    training_data.drop("etaRaw", axis=1, inplace=True)
    training_data.drop("portId", axis=1, inplace=True)
    training_data.drop("navstat", axis=1, inplace=True)
    training_data.drop("sog", axis=1, inplace=True)
    training_data.drop("cog", axis=1, inplace=True)
    training_data.drop("heading", axis=1, inplace=True)
    training_data.drop("rot", axis=1, inplace=True)

    # Calculate time difference to the next row
    training_data["time"] = pd.to_datetime(training_data["time"])
    training_data["time_diff"] = (
        -training_data.groupby("vesselId")["time"].diff(-1).dt.total_seconds()
    )

    # Reorder columns to place latitude and longitude after time
    columns = list(training_data.columns)
    time_index = columns.index("time")
    lat_lon_columns = ["latitude", "longitude"]
    for col in lat_lon_columns:
        columns.remove(col)
    for i, col in enumerate(lat_lon_columns):
        columns.insert(time_index + 1 + i, col)
    training_data = training_data[columns]

    return training_data


def features_and_labels(
    training_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features and labels from the training data
    :param data: data to extract features and labels from
    :return: features and labels
    """
    features = (
        training_data.groupby("vesselId")
        .apply(lambda x: x.iloc[:-1])
        .reset_index(drop=True)
    )
    
    # Haakons old.
    features.drop("vesselId", axis=1, inplace=True)
    features.drop("time", axis=1, inplace=True)
    
    # Edvards new.
    # features.drop("time_diff", axis=1, inplace=True)
    
    labels = (
        training_data.groupby("vesselId")
        .apply(lambda x: x.iloc[1:])
        .reset_index(drop=True)
    )
    labels.drop("vesselId", axis=1, inplace=True)
    labels.drop("time", axis=1, inplace=True)
    labels.drop("time_diff", axis=1, inplace=True)

    return features, labels

# New as of 22. oktober.
 
def pre_process_test_data_2(test_data: pd.DataFrame, preprocessed_vessel_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the test data, integrating the previous known lat/long for each vesselId only for the first instance,
    and sorting the combined dataframe by vesselId and time_diff.
    :param test_data: test data to be pre-processed
    :param preprocessed_vessel_data: pre-processed vessel data with known lat/long
    :return: pre-processed test data with previous_lat and previous_long integrated
    """
    # Convert 'time' column to datetime format
    test_data['time'] = pd.to_datetime(test_data['time'])

    # Calculate 'time_diff' as the difference from the last signal sent for each vessel
    test_data['time_diff'] = test_data.groupby('vesselId')['time'].diff().dt.total_seconds()

    # Fill any NaNs in 'time_diff' with 0 (for the first entry of each vessel)
    test_data['time_diff'].fillna(0, inplace=True)

    # Add placeholder columns for 'latitude' and 'longitude' since the model will predict these
    test_data['latitude'] = 0.0  # or use NaN if preferred
    test_data['longitude'] = 0.0  # or use NaN if preferred

    # Find the last known latitude and longitude from preprocessed_vessel_data for each vesselId
    last_known_positions = preprocessed_vessel_data.groupby('vesselId').last()[['latitude', 'longitude']].reset_index()
    last_known_positions = last_known_positions.rename(columns={'latitude': 'previous_lat', 'longitude': 'previous_long'})

    # Merge the test data with the last known positions
    test_data = pd.merge(test_data, last_known_positions, on='vesselId', how='left')

    # Sort the dataframe by 'vesselId' and 'time_diff'
    test_data = test_data.sort_values(by=['vesselId', 'time_diff']).reset_index(drop=True)

    # Create a mask for the first occurrence of each vesselId
    first_occurrence_mask = test_data.groupby('vesselId').cumcount() == 0

    # Set previous_lat and previous_long only for the first occurrence of each vesselId, leave the rest as NaN
    test_data.loc[~first_occurrence_mask, ['previous_lat', 'previous_long']] = 0, 0

    # Ensure the ID column stays in the original position
    id_column = test_data['ID']
    test_data = test_data.drop(columns=['ID'])
    test_data.insert(0, 'ID', id_column)

    # Drop unnecessary columns like 'time' and 'scaling_factor'
    test_data = test_data.drop(columns=['time', 'scaling_factor'], errors='ignore')

    return test_data





# New as of 22. okt

def preprocess_vessel_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the vessel data to extract relevant features: time_diff, longitude, latitude, 
    previous_long, previous_lat.
    
    :param data: DataFrame containing the vessel data
    :return: Pre-processed DataFrame
    """
    # Step 1: Convert 'time' column to datetime format
    data['time'] = pd.to_datetime(data['time'])
    
    # Step 2: Sort the data by 'vesselId' and 'time' to ensure proper ordering for each vessel
    data = data.sort_values(by=['vesselId', 'time'])
    
    # Step 3: Group by 'vesselId' to compute time difference and previous positions for each vessel
    data['time_diff'] = data.groupby('vesselId')['time'].diff().dt.total_seconds()  # Time difference in seconds
    
    # Step 4: Extract the previous longitude and latitude for each vessel
    data['previous_long'] = data.groupby('vesselId')['longitude'].shift(1)
    data['previous_lat'] = data.groupby('vesselId')['latitude'].shift(1)
    
    # Step 5: Fill missing values in time_diff with 0 (for the first entry of each vessel)
    data['time_diff'].fillna(0, inplace=True)
    
    # Step 6: Drop rows where the previous_lat or previous_long is NaN (first row for each vessel)
    data = data.dropna(subset=['previous_long', 'previous_lat'])
    
    # Step 7: Select only the relevant columns for the model
    data = data[['time', 'time_diff', 'longitude', 'latitude', 'previous_long', 'previous_lat', 'vesselId']]
    
    return data


if __name__ == "__main__":
    # training_data = pd.read_csv("task/ais_train.csv", delimiter="|")
    # training_data = pre_process(training_data)
    # training_data.to_csv("data/training_data_preprocessed.csv", index=False)
    
    training_data = pd.read_csv("task/ais_train.csv", delimiter="|")
    training_data = preprocess_vessel_data(training_data)
    training_data.to_csv("data/preprocessed_vessel_data.csv", index=False)
    features, labels = features_and_labels(training_data)
    features.to_csv("data/features.csv", index=False)
    labels.to_csv("data/labels.csv", index=False)
    test_data = pd.read_csv("task/ais_test.csv")
    test_data = pre_process_test_data_2(test_data, training_data)
    test_data.to_csv("data/merged_test_and_train_data.csv", index=False)


