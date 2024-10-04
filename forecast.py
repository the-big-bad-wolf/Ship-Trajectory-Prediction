import pandas as pd
import torch
from lstm import LSTMModel


def feature_engineer_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    # Convert the time column to datetime
    train_df["time"] = pd.to_datetime(train_df["time"])
    test_df["time"] = pd.to_datetime(test_df["time"])

    # Calculate the time difference between the last time in the training set and the time of each vessel in the test set
    # Get the latest time and features for each vesselId in the training set
    latest_time_train = train_df.groupby("vesselId")["time"].max().reset_index()
    latest_time_train = pd.merge(
        latest_time_train, train_df, on=["vesselId", "time"], how="left"
    )
    latest_time_train.rename(columns={"time": "latest_time"}, inplace=True)

    # Merge the latest time into the test set based on vesselId
    test_df_temp = pd.merge(
        test_df.drop_duplicates(subset="vesselId", keep="first"),
        latest_time_train,
        on="vesselId",
        how="left",
    )

    # Step 4: Calculate the time difference
    test_df_temp["time_diff"] = (
        test_df_temp["latest_time"] - test_df_temp["time"]
    ).dt.total_seconds()

    # Calculate time difference within each vesselId group
    test_df["time_diff"] = test_df.groupby("vesselId")["time"].diff().dt.total_seconds()
    # If time_diff is NA, calculate time_diff based on latest_time
    test_df["time_diff"].fillna(test_df_temp["time_diff"], inplace=True)

    # Drop the temporary column 'latest_time'
    test_df_temp.drop("latest_time", axis=1, inplace=True)

    # Join test_df_1 and test_df on 'vesselId' and 'time'
    test_df = pd.merge(
        test_df,
        test_df_temp,
        on=["vesselId", "time"],
        how="left",
        suffixes=("", "_duplicate"),
    )
    test_df.drop(
        [col for col in test_df.columns if "duplicate" in col], axis=1, inplace=True
    )
    # Move the 'time_diff' column to the end
    time_diff = test_df.pop("time_diff")
    test_df["time_diff"] = time_diff
    return test_df


# Load the model
model = LSTMModel(input_size=22, hidden_size=50, num_layers=1, output_size=2)
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

# Load the training data
train_df = pd.read_csv("data_preprocessed.csv")

# Load the test data
test_df = pd.read_csv("task/ais_test.csv")

# feature engineer the test data
test_df = feature_engineer_test(train_df, test_df)
test_df.to_csv("test_data_engineered.csv", index=False)


# Predict the next location for each vessel
vessel_groups = test_df.groupby("vesselId")
predictions = []

for vesselId, group in vessel_groups:
    temp = group.drop(["vesselId", "time", "ID", "scaling_factor"], axis=1).to_numpy()
    for i in range(1, temp.shape[0]):
        temp[i, :-1] = temp[i - 1, :-1]
    temp = temp.astype("float32")
    features = torch.tensor(temp)
    for i in range(0, features.shape[0]):
        prediction = model(features[i].unsqueeze(0).unsqueeze(0))
        predictions.append(
            {
                "ID": group.iloc[i]["ID"],
                "latitude": prediction[0, 0].item(),
                "longitude": prediction[0, 1].item(),
            }
        )
        features[i, 4] = prediction[0, 0]
        features[i, 5] = prediction[0, 1]

    # Save the predictions
    predictions_df = pd.DataFrame(predictions, columns=["latitude", "longitude", "ID"])
    predictions_df = predictions_df[["ID", "latitude", "longitude"]]
    predictions_df.to_csv("predictions.csv", index=False)
