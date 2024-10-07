from calendar import c
import pandas as pd

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

nan_values = labels.isna().sum()
print("Number of NaN values in each column:")
print(nan_values)
