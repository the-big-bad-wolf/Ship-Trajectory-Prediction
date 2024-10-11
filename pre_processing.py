import pandas as pd


def pre_process(training_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processing of the data
    :param data: data to be pre-processed
    :return: pre-processed data
    """
    training_data.drop("etaRaw", axis=1, inplace=True)
    training_data.drop("portId", axis=1, inplace=True)

    # Convert navstat to binary anchor feature
    training_data["navstat"] = training_data["navstat"].apply(
        lambda x: 1 if (x in [1, 5, 6]) else 0
    )
    training_data.rename(columns={"navstat": "anchored"}, inplace=True)

    # Set sog to 0 if it is 102.3 and the vessel is anchored, otherwise interpolate
    training_data["sog"] = training_data.apply(
        lambda row: 0 if row["sog"] == 102.3 and row["anchored"] == 1 else row["sog"],
        axis=1,
    )
    training_data["sog"] = (
        training_data["sog"]
        .mask((training_data["sog"] == 102.3) & (training_data["anchored"] == 0))
        .interpolate(limit_area="inside", limit=1)
    )

    # Normalize the SOG feature
    training_data["sog"] = training_data["sog"] / 102.2

    # Replace heading with COG if heading is out of bounds and vice versa
    training_data["heading"] = training_data.apply(
        lambda row: row["cog"] if row["heading"] >= 360 else row["heading"], axis=1
    )
    training_data["cog"] = training_data.apply(
        lambda row: row["heading"] if row["cog"] >= 360 else row["cog"], axis=1
    )

    # Fill in COG and heading with the previous or next row if both are over 360
    training_data["cog"] = (
        training_data["cog"]
        .mask(training_data["cog"] >= 360)
        .ffill(limit=1)
        .bfill(limit=1)
    )
    training_data["heading"] = (
        training_data["heading"]
        .mask(training_data["heading"] >= 360)
        .ffill(limit=1)
        .bfill(limit=1)
    )

    # Normalize the heading and COG feature
    training_data["cog"] = training_data["cog"] / 360
    training_data["heading"] = training_data["heading"] / 360

    # Set ROT to 0 if out of no ROT sensor is available
    training_data["rot"] = training_data["rot"].apply(lambda x: 0 if x == 128 else x)

    # Interpolate ROT for values at -127 or 127
    training_data["rot"] = (
        training_data["rot"]
        .mask(training_data["rot"].isin([-127, 127]))
        .interpolate(limit_area="inside", limit=3)
    )

    # Normalize the ROT feature
    training_data["rot"] = training_data["rot"] / 126

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
    # features.drop("vesselId", axis=1, inplace=True)
    # features.drop("time", axis=1, inplace=True)
    
    # Edvards new.
    features.drop("time_diff", axis=1, inplace=True)
    
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
    training_data.to_csv("data/training_data_preprocessed.csv", index=False)
    features, labels = features_and_labels(training_data)
    features.to_csv("data/features.csv", index=False)
    labels.to_csv("data/labels.csv", index=False)
