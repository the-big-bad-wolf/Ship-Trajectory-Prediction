import pandas as pd
import torch
from neural_network import ShipTrajectoryMLP


def create_test_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    # Convert the time column to datetime
    train_df["time"] = pd.to_datetime(train_df["time"])
    test_df["time"] = pd.to_datetime(test_df["time"])

    latest_time_train = train_df.loc[train_df.groupby("vesselId")["time"].idxmax()]
    latest_time_train.rename(columns={"time": "latest_time"}, inplace=True)

    # Merge the latest time from the training data with the test data
    test_df = pd.merge(
        test_df,
        latest_time_train,
        on=["vesselId"],
        how="left",
    )
    # Calculate time difference within each vesselId group
    test_df["time_diff"] = test_df.groupby("vesselId")["time"].diff().dt.total_seconds()
    # If time_diff is NA, calculate time_diff based on latest_time
    test_df["time_diff"].fillna(
        (test_df["time"] - test_df["latest_time"]).dt.total_seconds(),
        inplace=True,
    )

    # Drop unused columns
    test_df.drop(["latest_time"], axis=1, inplace=True)

    return test_df


# Load the training data
train_df = pd.read_csv("data/training_data_preprocessed.csv")

# Load the test data
test_df = pd.read_csv("task/ais_test.csv")

# Create the test features
test_features = create_test_features(train_df, test_df)
test_features.to_csv("data/test_features.csv", index=False)

# Predict the next location for each vessel
vessel_groups = test_features.groupby("vesselId")
predictions = []

# Load the model
model = ShipTrajectoryMLP(test_features.shape[1] - 4, 12, test_features.shape[1] - 5)
model.load_state_dict(torch.load("models/mlp_model_epoch_100.pth"))
model.eval()

for vesselId, group in vessel_groups:
    features = torch.tensor(
        group.drop(
            [
                "vesselId",
                "time",
                "ID",
                "scaling_factor",
            ],
            axis=1,
        )
        .to_numpy()
        .astype("float32")
    )
    for i in range(0, features.shape[0]):
        prediction = model(features[i].unsqueeze(0))
        prediction = prediction[0]
        predictions.append(
            {
                "ID": group.iloc[i]["ID"],
                "longitude": prediction[1].item(),
                "latitude": prediction[0].item(),
            }
        )
        if i < features.shape[0] - 1:
            features[i + 1, :-1] = prediction[:]


# Save the predictions
predictions_df = pd.DataFrame(predictions, columns=["ID", "longitude", "latitude"])
predictions_df.rename(
    columns={"latitude": "latitude_predicted", "longitude": "longitude_predicted"},
    inplace=True,
)
predictions_df.sort_values("ID", inplace=True)
predictions_df.to_csv("predictions.csv", index=False)

# Print the number of NaN values in the predicted features
print(predictions_df.isnull().sum())
