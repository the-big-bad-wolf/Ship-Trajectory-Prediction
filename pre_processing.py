import pandas as pd


def pre_process(training_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processing of the data
    :param data: data to be pre-processed
    :return: pre-processed data
    """
    training_data.drop("etaRaw", axis=1, inplace=True)
    training_data.drop("portId", axis=1, inplace=True)

    navstat_dummies = pd.get_dummies(training_data["navstat"], prefix="navstat")
    training_data = pd.concat([training_data, navstat_dummies], axis=1)
    training_data.drop("navstat", axis=1, inplace=True)

    training_data["time"] = pd.to_datetime(training_data["time"])
    # Calculate time difference to the next row
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
    features.drop("vesselId", axis=1, inplace=True)
    features.drop("time", axis=1, inplace=True)

    labels = (
        training_data.groupby("vesselId")
        .apply(lambda x: x.iloc[1:])
        .reset_index(drop=True)
    )
    labels.drop("vesselId", axis=1, inplace=True)
    labels.drop("time", axis=1, inplace=True)
    labels.drop("time_diff", axis=1, inplace=True)

    return features, labels


if __name__ == "__main__":
    training_data = pd.read_csv("task/ais_train.csv", delimiter="|")
    training_data = pre_process(training_data)
    training_data.to_csv("training_data_preprocessed.csv", index=False)
    features, labels = features_and_labels(training_data)
    features.to_csv("features.csv", index=False)
    labels.to_csv("labels.csv", index=False)
