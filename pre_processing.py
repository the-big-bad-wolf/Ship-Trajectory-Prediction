#from global_land_mask import globe

def pre_process(train_df):

     # Convert navstat to binary anchor feature
    train_df["moored"] = train_df["navstat"].apply(
        lambda x: 1 if (x in [1, 5, 6]) else 0
    )

    train_df.loc[(train_df["sog"] == 102.3) & (train_df["moored"] == 1), "sog"] = 0

    train_df['sog'] = train_df.groupby('vesselId')['sog'].transform(
    lambda group: group.mask(group >= 102.3).interpolate().ffill().bfill()
    )

    # Normalize the SOG feature
    train_df["sog"] = train_df["sog"] / 102.2

    # Interpolate the COG value between the previous and next value per vesselId if it is 360 or above
    train_df['cog'] = train_df.groupby('vesselId')['cog'].transform(
    lambda group: group.mask(group >= 360).interpolate().ffill().bfill()
    )

    # Interpolate the heading value between the previous and next value per vesselId if it is 360 or above
    train_df['heading'] = train_df.groupby('vesselId')['heading'].transform(
    lambda group: group.mask(group >= 360).interpolate().ffill().bfill()
    )

    # Normalize the heading and COG feature
    train_df["cog"] = train_df["cog"] / 360
    train_df["heading"] = train_df["heading"] / 360

    return train_df


def add_vessel_type(vessel_df, df):
    
    vessel_subset = vessel_df.loc[:, ['shippingLineId', 'vesselId', 'length']]

    # Add the vessel type feature to the dataframe
    df_total = df.merge(vessel_subset, on='vesselId', how='left')

    # Add deep-sea boat feature. If length > 200, assign 1, else 0
    df_total['deep_sea'] = df_total['length'].apply(lambda x: 1 if x > 200 else 0)

    # Drop the length column
    #df_total.drop('length', axis=1, inplace=True)

    return df_total


# def resample(df, delta_t):

#     # Interpolate the time_diff feature such that we have values every delta_t seconds


#     return df