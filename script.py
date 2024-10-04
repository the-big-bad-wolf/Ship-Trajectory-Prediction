import pandas as pd

# Load the CSV file
df = pd.read_csv("predictions.csv")

# Fill missing values with the previous row
df.fillna(method="ffill", inplace=True)

# Sort the DataFrame by the 'ID' column
df_sorted = df.sort_values(by="ID")

# Switch the order of 'longitude' and 'latitude' columns
df_sorted = df_sorted[["ID", "longitude_predicted", "latitude_predicted"]]

# Save the sorted DataFrame back to a CSV file
df_sorted.to_csv("predictions_sorted.csv", index=False)
