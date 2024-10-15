from calendar import c
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

labels = pd.read_csv("data/labels.csv")

max_heading = labels["heading"].max()
max_cog = labels["cog"].max()
print(f"The maximum value for heading is: {max_heading}")
print(f"The maximum value for COG is: {max_cog}")

count_heading_over_360 = (labels["heading"] >= 1).sum()
count_cog_over_360 = (labels["cog"] >= 1).sum()
print(f"Number of headings over 360: {count_heading_over_360}")
print(f"Number of COGs over 360: {count_cog_over_360}")


count_both_over_360 = ((labels["heading"] >= 1) & (labels["cog"] >= 1)).sum()
print(f"Number of rows with both heading and COG over 360: {count_both_over_360}")

count_heading_over_360_not_cog = ((labels["heading"] >= 1) & (labels["cog"] < 1)).sum()
print(
    f"Number of rows with heading over 360 but not COG: {count_heading_over_360_not_cog}"
)


count_cog_over_360_not_heading = ((labels["heading"] < 1) & (labels["cog"] >= 1)).sum()
print(
    f"Number of rows with COG over 360 but not heading: {count_cog_over_360_not_heading}"
)


max_sog = labels["sog"].max()
print(f"The maximum value for SOG is: {max_sog}")

count_sog_max = (labels["sog"] == max_sog).sum()
print(f"Number of SOGs at max: {count_sog_max}")

min_rot = labels["rot"].min()
max_rot = labels["rot"].max()
print(f"The minimum value for ROT is: {min_rot}")
print(f"The maximum value for ROT is: {max_rot}")

count_min_rot = (labels["rot"] == min_rot).sum()
count_max_rot = (labels["rot"] == max_rot).sum()
print(f"Number of ROTs at min: {count_min_rot}")
print(f"Number of ROTs at max: {count_max_rot}")

max_latitude = labels["latitude"].max()
print(f"The maximum value for latitude is: {max_latitude}")

min_latitude = labels["latitude"].min()
print(f"The minimum value for latitude is: {min_latitude}")

max_longitude = labels["longitude"].max()
max_longitude_row = labels[labels["longitude"] == max_longitude].iloc[0]
print(
    f"The maximum value for longitude is: {max_longitude} at coordinates ({max_longitude_row['latitude']}, {max_longitude_row['longitude']})"
)

min_longitude = labels["longitude"].min()
min_longitude_row = labels[labels["longitude"] == min_longitude].iloc[0]
print(
    f"The minimum value for longitude is: {min_longitude} at coordinates ({min_longitude_row['latitude']}, {min_longitude_row['longitude']})"
)
print(f"The maximum value for longitude is: {max_longitude}")

min_longitude = labels["longitude"].min()
print(f"The minimum value for longitude is: {min_longitude}")

nan_values = labels.isna().sum()
print("Number of NaN values in each column:")
print(nan_values)


import matplotlib.pyplot as plt

world_map = gpd.read_file("task/map/ne_110m_admin_0_countries.shp")
data = pd.read_csv("task/ais_train.csv", delimiter="|")

min_longitude = data["longitude"].min()
min_longitude_row = data[data["longitude"] == min_longitude].iloc[0]
min_vessel_id = min_longitude_row["vesselId"]
vessel_data = data[data["vesselId"] == min_vessel_id]
vessel_data = vessel_data.sort_values(by="time")

geometry = [Point(xy) for xy in zip(vessel_data["longitude"], vessel_data["latitude"])]
geo_df = gpd.GeoDataFrame(vessel_data, geometry=geometry)

fig, ax = plt.subplots(figsize=(15, 15))
world_map.plot(ax=ax, color="lightgrey")
geo_df.plot(ax=ax, marker="o", color="red", markersize=5)
plt.title(f"Path of vessel with ID {min_vessel_id}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
