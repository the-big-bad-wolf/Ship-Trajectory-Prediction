import pandas as pd


def pre_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processing of the data
    :param data: data to be pre-processed
    :return: pre-processed data
    """
    data.sort_values(by=["vesselId", "time"], inplace=True)

    data.drop("etaRaw", axis=1, inplace=True)
    data.drop("portId", axis=1, inplace=True)

    navstat_dummies = pd.get_dummies(data["navstat"], prefix="navstat")
    data = pd.concat([data, navstat_dummies], axis=1)
    data.drop("navstat", axis=1, inplace=True)

    data["time"] = pd.to_datetime(data["time"])
    # Calculate time difference to the next row
    data["time_diff"] = data.groupby("vesselId")["time"].diff(-1).dt.total_seconds()

    data.set_index("time", inplace=True)
    return data


def features_and_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features and labels from the data
    :param data: data to extract features and labels from
    :return: features and labels
    """
    features = (
        data.groupby("vesselId").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    )
    features.drop("vesselId", axis=1, inplace=True)

    labels = data.groupby("vesselId").apply(lambda x: x.iloc[1:]).reset_index(drop=True)
    labels = labels[["latitude", "longitude"]]
    return features, labels


if __name__ == "__main__":
    data = pd.read_csv("task/ais_train.csv", delimiter="|")
    data = pre_processing(data)
    data.to_csv("data_preprocessed.csv", index=False)
    features, labels = features_and_labels(data)
    features.to_csv("features.csv", index=False)
    labels.to_csv("labels.csv", index=False)
