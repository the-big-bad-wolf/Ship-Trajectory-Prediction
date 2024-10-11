from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd


# Load the data as a regular DataFrame first to inspect it
features = pd.read_csv("/Users/edvardschwabe/Ship-Trajectory-Prediction/data/features.csv")
labels = pd.read_csv("/Users/edvardschwabe/Ship-Trajectory-Prediction/data/labels.csv")
joined_data = features.join(labels, rsuffix="_target")# pd.join(features, labels)

joined_data.to_csv("data/joined_data.csv", index=False)
# Drop multiple columns by passing them as a list to `drop`
joined_data_long = joined_data.drop(["anchored_target", "heading_target", "rot_target", "sog_target", "cog_target", "latitude_target"], axis=1)
joined_data_lat = joined_data.drop(["anchored_target", "heading_target", "rot_target", "sog_target", "cog_target", "longitude_target"], axis=1)



joined_data_long.to_csv("data/joined_data_long.csv", index=False)
joined_data_lat.to_csv("data/joined_data_lat.csv", index=False)


joined_data['time'] = pd.to_datetime(joined_data['time'])

# Sort the DataFrame by 'vesselID' and 'time'
# joined_data_sorted = joined_data.sort_values(by=['vesselId', 'time'])


train_data_lat = TimeSeriesDataFrame.from_data_frame(
    joined_data_lat,
    id_column="vesselId",
    timestamp_column="time"
)

train_data_long = TimeSeriesDataFrame.from_data_frame(
    joined_data_long,
    id_column="vesselId",
    timestamp_column="time"
)


lat_mod = TimeSeriesPredictor.load("latitude_model")
long_mod = TimeSeriesPredictor.load("longitude_model")

predictions_lat = lat_mod.predict(train_data_lat)
predictions_long = long_mod.predict(train_data_long)
predictions_lat.to_csv("pred_lat")
predictions_long.to_csv("pred_long")

# train_data.head()

# print(data.head())


pred_lat = TimeSeriesPredictor(
    prediction_length=120,
    path="latitude_model",
    target="latitude_target",
    eval_metric="MASE",
    freq='H',
)

pred_lat.fit(
    train_data_lat,
    presets="medium_quality",
    time_limit=600,
)

pred_long = TimeSeriesPredictor(
    prediction_length=120,
    path="longitude_model",
    target="longitude_target",
    eval_metric="MASE",
    freq='H',
)

pred_long.fit(
    train_data_long,
    presets="medium_quality",
    time_limit=600,
)


predictions_lat = lat_mod.predict(train_data_lat)
predictions_long = long_mod.predict(train_data_long)
predictions_lat.to_csv("pred_lat")
predictions_long.to_csv("pred_long")