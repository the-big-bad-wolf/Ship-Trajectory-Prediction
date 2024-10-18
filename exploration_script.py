import pandas as pd
from shapely.geometry import Point, LineString
import geopandas as gpd

original = pd.read_csv("task/ais_train.csv", delimiter="|")
vessels_df = pd.read_csv("task/vessels.csv", delimiter="|")
processed_data = pd.read_csv("data/training_data_preprocessed.csv")

# Print the number of unique vesselIds
unique_vessel_ids = original["vesselId"].nunique()
print(f"Number of unique vesselIds: {unique_vessel_ids}")

# Print the number of NaN values in each column
nan_counts = processed_data.isna().sum()
print("NaN values in each column in pre-processed training file:")
print(nan_counts)

# Load the vessels.csv file

# Print the number of NaN values in each column
nan_counts_vessels = vessels_df.isna().sum()
print("NaN values in each column in vessels file:")
print(nan_counts_vessels)

# Print the maximum SOG (Speed Over Ground)
max_sog = processed_data["sog"].max()
print(f"Maximum SOG: {max_sog}")

# Print the maximum SOG (Speed Over Ground) when anchor is 1
max_sog_anchor_1 = processed_data[processed_data["anchored"] == 1]["sog"].max()
print(f"Maximum SOG when anchor is 1: {max_sog_anchor_1}")

# Print the average SOG (Speed Over Ground) when anchor is 1
average_sog_anchor_1 = processed_data[processed_data["anchored"] == 1]["sog"].mean()
print(f"Average SOG when anchor is 1: {average_sog_anchor_1}")

# Find the vessels with the 10 fewest data points
vessel_counts = processed_data["vesselId"].value_counts().nsmallest(10)

# Print the highest and lowest latitude in the dataset
highest_latitude = processed_data["latitude"].max()
lowest_latitude = processed_data["latitude"].min()
print(f"Highest latitude: {highest_latitude}")
print(f"Lowest latitude: {lowest_latitude}")

# Print the vessels and their data point counts
print("Vessel ID - Data Points")
for vessel_id, count in vessel_counts.items():
    print(f"{vessel_id} - {count}")


import matplotlib.pyplot as plt

# Filter ships by length
df_below_193 = processed_data[processed_data["length"] < 160 / 300]
df_above_193 = processed_data[processed_data["length"] >= 160 / 300]

# Select 5 random ships from each group
random_ships_below_193 = df_below_193["vesselId"].drop_duplicates().sample(5)
random_ships_above_193 = df_above_193["vesselId"].drop_duplicates().sample(5)

# Plotting
fig, ax = plt.subplots(figsize=(15, 10))

# Plot paths for ships with length below 193 in red
for ship_id in random_ships_below_193:
    ship_data = df_below_193[df_below_193["vesselId"] == ship_id]
    points = [Point(xy) for xy in zip(ship_data["longitude"], ship_data["latitude"])]
    line = LineString(points)
    gpd.GeoSeries([line]).plot(ax=ax, label=f"Ship {ship_id} (below 193)", color="red")

# Plot paths for ships with length above 193
for ship_id in random_ships_above_193:
    ship_data = df_above_193[df_above_193["vesselId"] == ship_id]
    points = [Point(xy) for xy in zip(ship_data["longitude"], ship_data["latitude"])]
    line = LineString(points)
    gpd.GeoSeries([line]).plot(ax=ax, label=f"Ship {ship_id} (above 193)")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Paths of Random Ships")
plt.show()
