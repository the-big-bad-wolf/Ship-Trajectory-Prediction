from geopy.distance import geodesic

def prepare_output(predictions, df):
    output_df = df.copy()
    output_df['longitude_predicted'] = predictions[:, 0]
    output_df['latitude_predicted'] = predictions[:, 1]
    return output_df

def calculate_distance(row):
    # Calculate the weighted distance for each row
    distance = geodesic((row['latitude'], row['longitude']), 
                        (row['latitude_predicted'], row['longitude_predicted'])).meters
    # Weight the distance by the scaling factor
    weighted_distance = distance * 0.3
    return weighted_distance

def calculate_score(solution_submission):
        # Calculate the weighted distance for each row
        solution_submission['weighted_distance'] = solution_submission.apply(calculate_distance, axis=1)

        weighted_distance = solution_submission['weighted_distance'].mean() / 1000.0

        return weighted_distance