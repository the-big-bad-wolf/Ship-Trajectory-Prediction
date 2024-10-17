import pandas as pd


def impute(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the data
    :param data: data to impute missing values
    :return: data with imputed missing values
    """
    # Set sog to 0 if it is 102.3 and the vessel is anchored, otherwise interpolate
    data.loc[(data["sog"] == 102.3) & (data["anchored"] == 1), "sog"] = 0
    data["sog"] = data["sog"].replace(102.3, float("nan"))
    data["sog"] = (
        data.groupby("vesselId", as_index=False)
        .apply(lambda group: group["sog"].interpolate(limit=3))
        .reset_index(drop=True)
    )

    # Replace heading with COG if heading is out of bounds and vice versa
    data["heading"] = data.apply(
        lambda row: row["cog"] if row["heading"] >= 360 else row["heading"], axis=1
    )
    data["cog"] = data.apply(
        lambda row: row["heading"] if row["cog"] >= 360 else row["cog"], axis=1
    )

    # Fill in COG and heading with the previous or next row if both are over 360
    data["cog"] = (
        data.groupby("vesselId", as_index=False)
        .apply(lambda group: group["cog"].mask(group["cog"] >= 360).ffill().bfill())
        .reset_index(drop=True)
    )
    data["heading"] = (
        data.groupby("vesselId", as_index=False)
        .apply(
            lambda group: group["heading"].mask(group["heading"] >= 360).ffill().bfill()
        )
        .reset_index(drop=True)
    )

    # Interpolate ROT for values at -127 or 127 or 128
    data["rot"] = (
        data.groupby("vesselId", as_index=False)
        .apply(
            lambda group: group["rot"]
            .mask(group["rot"].isin([-127, 127, 128]))
            .interpolate(limit_area="inside", limit=1)
            .fillna(0)
        )
        .reset_index(drop=True)
    )

    return data


def standardize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the data
    :param data: data to be standardized
    :return: standardized data
    """
    columns_to_standardize = [
        col
        for col in data.columns
        if col
        not in [
            "latitude",
            "longitude",
            "time",
            "vesselId",
            "deep_sea",
            "anchored",
            "restricted",
        ]
    ]
    data[columns_to_standardize] = (
        data[columns_to_standardize] - data[columns_to_standardize].mean()
    ) / data[columns_to_standardize].std()
    return data


def pre_process(training_data: pd.DataFrame, vessel_data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processing of the data
    :param data: data to be pre-processed
    :return: pre-processed data
    """
    training_data["time"] = pd.to_datetime(training_data["time"])
    training_data.sort_values(by=["vesselId", "time"], inplace=True)
    training_data.reset_index(drop=True, inplace=True)

    training_data.drop("etaRaw", axis=1, inplace=True)
    training_data.drop("portId", axis=1, inplace=True)

    vessel_data = vessel_data[["vesselId", "length", "CEU", "GT", "yearBuilt"]]
    training_data = training_data.merge(vessel_data, on="vesselId", how="left")

    # Create a new feature indicating whether the ship is longer than 160 meters
    training_data["deep_sea"] = training_data["length"].apply(
        lambda x: 1 if x >= 160 else 0
    )
    training_data["length"] = training_data["length"] / 300

    # Convert navstat to binary anchor and restricted features
    training_data["anchored"] = training_data["navstat"].apply(
        lambda x: 1 if (x in [1, 5, 6]) else 0
    )
    training_data["restricted"] = training_data["navstat"].apply(
        lambda x: 1 if (x in [2, 3, 4, 7, 14]) else 0
    )
    training_data.drop("navstat", axis=1, inplace=True)

    # Calculate time difference to the next row
    training_data["time_diff"] = (
        -training_data.groupby("vesselId")["time"].diff(-1).dt.total_seconds()
    )

    # Impute missing values
    training_data = impute(training_data)

    # Standardize the data
    training_data = standardize(training_data)

    # Reorder columns to make vesselId, time, latitude, longitude the first columns
    columns_order = ["vesselId", "time", "latitude", "longitude"] + [
        col
        for col in training_data.columns
        if col not in ["vesselId", "time", "latitude", "longitude"]
    ]
    training_data = training_data[columns_order]

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
    AIS_data = pd.read_csv("task/ais_train.csv", delimiter="|")
    vessel_data = pd.read_csv("task/vessels.csv", delimiter="|")
    training_data = pre_process(AIS_data, vessel_data)
    training_data.to_csv("data/training_data_preprocessed.csv", index=False)
    features, labels = features_and_labels(training_data)
    features.to_csv("data/features.csv", index=False)
    labels.to_csv("data/labels.csv", index=False)
