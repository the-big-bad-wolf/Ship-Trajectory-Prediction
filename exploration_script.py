import pandas as pd
from shapely.geometry import Point, LineString
import geopandas as gpd
import random

df = pd.read_csv("data/training_data_preprocessed.csv")


import matplotlib.pyplot as plt

# Filter ships by length
df_below_193 = df[df["length"] < 160]
df_above_193 = df[df["length"] >= 160]

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
