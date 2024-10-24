import pandas as pd
import numpy as np

def create_features_and_labels_dataFrame(
    training_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features and labels from the training data
    :param training_data: data to extract features and labels from
    :return: features and labels
    """
    
    # Extract features: take all columns except the ones we want to predict (latitude, longitude, time)
    features = (
        training_data.groupby("vesselId")
        .apply(lambda x: x.iloc[:-1])  # Use the past data as features
        .reset_index(drop=True)
    )
    
    # Remove irrelevant columns from features
    # features.drop(columns=["vesselId"], inplace=True)
    
    # Extract labels: the target values for prediction, i.e., the next latitude and longitude
    labels = (
        training_data.groupby("vesselId")
        .apply(lambda x: x.iloc[1:])  # Use the next row as the label for each current row
        .reset_index(drop=True)
    )
    
    # Keep only latitude and longitude as labels (the targets to predict)
    labels = labels[["latitude", "longitude", "cog", "sog", "heading"]]
    
    return features, labels


# New as of 22. oktober.
 
def preprocess_AIS_test(test_data: pd.DataFrame, preprocessed_vessel_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the test data, integrating the previous known lat/long, sog, heading, and cog for each vesselId
    only for the first instance, and sorting the combined dataframe by vesselId and time_diff.
    
    :param test_data: test data to be pre-processed
    :param preprocessed_vessel_data: pre-processed vessel data with known lat/long, sog, heading, and cog
    :return: pre-processed test data with previous_lat, previous_long, sog, heading, and cog integrated
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

    # Find the last known latitude, longitude, sog, heading, and cog from preprocessed_vessel_data for each vesselId
    last_known_positions = preprocessed_vessel_data.groupby('vesselId').last()[['latitude', 'longitude', 'sog', 'heading', 'cog']].reset_index()
    
    # Rename the columns for clarity when merging
    last_known_positions = last_known_positions.rename(columns={
        'latitude': 'previous_lat', 
        'longitude': 'previous_long', 
        'sog': 'previous_sog', 
        'heading': 'previous_heading', 
        'cog': 'previous_cog'
    })

    # Merge the test data with the last known positions
    test_data = pd.merge(test_data, last_known_positions, on='vesselId', how='left')

    # Sort the dataframe by 'vesselId' and 'ID'
    test_data = test_data.sort_values(by=['vesselId', 'ID']).reset_index(drop=True)

    # Create a mask for the first occurrence of each vesselId
    first_occurrence_mask = test_data.groupby('vesselId').cumcount() == 0

    # Set previous_lat, previous_long, sog, heading, and cog only for the first occurrence of each vesselId
    # Leave the rest as 0 for now (since these will be dynamically updated during the row-by-row prediction)
    test_data.loc[~first_occurrence_mask, ['previous_lat', 'previous_long', 'previous_sog', 'previous_heading', 'previous_cog']] = 0, 0, 0, 0, 0

    # Ensure the ID column stays in the original position
    id_column = test_data['ID']
    test_data = test_data.drop(columns=['ID'])
    test_data.insert(0, 'ID', id_column)

    # Drop unnecessary columns like 'time' and 'scaling_factor'
    test_data = test_data.drop(columns=['time', 'scaling_factor'], errors='ignore')

    return test_data

# New as of 22. okt

def preprocess_AIS_train(data: pd.DataFrame, cutoff: bool) -> pd.DataFrame:
    """
    Pre-process the vessel data to extract relevant features, including standardization, interpolation,
    and cyclical transformations for heading and COG.
    
    :param data: DataFrame containing the vessel data
    :param cutoff: Boolean indicating whether to apply a cutoff to limit data to the last 2000 measurements for vessels
    :return: Pre-processed DataFrame
    """
    # Step 1: Convert 'time' column to datetime format
    data['time'] = pd.to_datetime(data['time'])
    
    # Step 2: Sort the data by 'vesselId' and 'time' to ensure proper ordering for each vessel
    data = data.sort_values(by=['vesselId', 'time'])
    
    if cutoff:
        # Apply cutoff to remove old entries for vessels with more than 2000 rows
        vessel_counts = data['vesselId'].value_counts()
        overrepresented_vessels = vessel_counts[vessel_counts > 2000].index
        data = data.groupby('vesselId').apply(lambda group: group.tail(2000) if group.name in overrepresented_vessels else group).reset_index(drop=True)
    
    # Step 3: Compute time differences and interpolate missing values
    data['time_diff'] = data.groupby('vesselId')['time'].diff().dt.total_seconds()
    data['time_diff'].fillna(0, inplace=True)

    # Set SOG to NaN where it's invalid (102.3) and interpolate missing values for SOG and COG
    data["sog"] = data["sog"].replace(102.3, np.nan)

    # Use transform instead of apply to maintain the index alignment
    data['sog'] = data.groupby('vesselId')['sog'].transform(lambda group: group.interpolate(limit=3))
    data['cog'] = data.groupby('vesselId')['cog'].transform(lambda group: group.mask(group >= 360).ffill().bfill())

    # Fix heading using COG where heading is invalid, and fill missing heading values
    data["heading"] = data.apply(lambda row: row["cog"] if row["heading"] >= 360 else row["heading"], axis=1)
    data['heading'] = data.groupby('vesselId')['heading'].transform(lambda group: group.ffill().bfill())

    # Step 4: Drop rows where longitude/latitude are out of bounds
    data = data[(data['longitude'].between(-180, 180)) & (data['latitude'].between(-90, 90))]
    
    # Step 5: Apply cos-sin transformation to COG and heading
    data['cog'] = np.sin(np.deg2rad(data['cog']))
    # data['cog_cos'] = np.cos(np.deg2rad(data['cog']))
    data['heading'] = np.sin(np.deg2rad(data['heading']))
    # data['heading_cos'] = np.cos(np.deg2rad(data['heading']))

    # Step 6: Extract previous states for each vessel
    data['previous_long'] = data.groupby('vesselId')['longitude'].shift(1)
    data['previous_lat'] = data.groupby('vesselId')['latitude'].shift(1)
    data['previous_sog'] = data.groupby('vesselId')['sog'].shift(1)
    data['previous_heading'] = data.groupby('vesselId')['heading'].shift(1)
    data['previous_cog'] = data.groupby('vesselId')['cog'].shift(1)
    
    # Step 7: Standardize numeric variables like time_diff, sog, etc.
    # This did not improve anything.
    # data['time_diff'] = (data['time_diff'] - data['time_diff'].mean()) / data['time_diff'].std()
    # data['sog'] = (data['sog'] - data['sog'].mean()) / data['sog'].std()

    # Step 8: Drop any rows with NaNs in the previous positions
    data = data.dropna(subset=['previous_long', 'previous_lat'])

    # Step 9: Select final features for the model
    data = data[['time', 'time_diff', 'longitude', 'latitude', 'previous_long', 'previous_lat', 
                 'previous_sog', 'previous_heading', 'previous_cog', 'vesselId', 'sog', 'heading', 'cog']]

    return data




if __name__ == "__main__":
    training_data = pd.read_csv("task/ais_train.csv", delimiter="|")
    training_data = preprocess_AIS_train(training_data, cutoff = True)
    training_data.to_csv("data/preprocessed_vessel_data.csv", index=False)
    features, labels = create_features_and_labels_dataFrame(training_data)
    features.to_csv("data/features.csv", index=False)
    labels.to_csv("data/labels.csv", index=False)
    test_data = pd.read_csv("task/ais_test.csv")
    test_data = preprocess_AIS_test(test_data, training_data)
    test_data.to_csv("data/merged_test_and_train_data.csv", index=False)


