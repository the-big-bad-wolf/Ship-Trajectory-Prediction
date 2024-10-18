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

def pre_process_test_data(test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the test data to match the format of training features.
    :param test_data: test data to be pre-processed
    :return: pre-processed test data
    """
    # Convert 'time' column to datetime format
    test_data['time'] = pd.to_datetime(test_data['time'])

    # Create a new feature 'time_diff' that measures time difference in seconds from the first timestamp
    reference_time = test_data['time'].min()
    test_data['time_diff'] = (test_data['time'] - reference_time).dt.total_seconds()

    # Drop 'time' and 'vesselId' columns since they are not needed for prediction
    test_data = test_data.drop(columns=['vesselId', 'scaling_factor', 'ID'])

    return test_data


if __name__ == "__main__":
    training_data = pd.read_csv("task/ais_train.csv", delimiter="|")
    training_data = pre_process(training_data)
    training_data.to_csv("data/training_data_preprocessed.csv", index=False)
    features, labels = features_and_labels(training_data)
    features.to_csv("data/features.csv", index=False)
    labels.to_csv("data/labels.csv", index=False)
    test_data = pd.read_csv("task/ais_test.csv")
    test_data = pre_process_test_data(test_data)
    test_data.to_csv("data/pre_processed_test_data", index=False)
