import pandas as pd
import numpy as np
import feature_engineering as fe
import validation as val
import pre_processing as pp

def process_data(train_csv, test_csv, vessel_csv):
    '''
    Process the raw data: pre process, feature engineering

    Args:
    training: str, path to the training data
    test: str, path to the test data

    '''
    # Load the data
    ais_train_df = pd.read_csv(train_csv, sep='|')
    ais_test_df = pd.read_csv(test_csv, sep=',')
    vessel_df = pd.read_csv(vessel_csv, sep='|')

    # Merge the vessel information
    ais_train_df = pp.add_vessel_type(vessel_df, ais_train_df)
    ais_test_df = pp.add_vessel_type(vessel_df, ais_test_df)

    # Assuming 'vesselId' is the categorical column with 688 unique entries
    ais_train_df, ais_test_df = fe.transform_categorical(ais_train_df, ais_test_df, 'vesselId')
    ais_train_df, ais_test_df = fe.transform_categorical(ais_train_df, ais_test_df, 'shippingLineId')

    # Feature engineering
    features_train_df = fe.feature_engineer_train(ais_train_df)
    features_test_df = fe.feature_engineer_test(ais_train_df, ais_test_df)

    return features_train_df, features_test_df

def prepare_data_to_train(features_train_df, features_test_df, features, split_train=False):
    '''
    Prepare the data for training and testing the model: split data and choose features

    Args:
    features_train_df: DataFrame, training data that are feature engineered
    features_test_df: DataFrame, test data that are feature engineered
    features: list, list of features to use for training the model
    split_train: bool, whether to split the training data into train and validation sets

    '''

    # Split the data into train and validation sets
    if split_train:
        train_df, validation_df = fe.split_train_validation(features_train_df)
        X_train, y_train = fe.choose_features(train_df, features, ['longitude', 'latitude']) 
        X_val, y_val = fe.choose_features(validation_df, features, ['longitude', 'latitude'])
        X_test, y_test = fe.choose_features(features_test_df, features, [])
        
        return X_train, y_train, X_val, y_val, validation_df, X_test, y_test
    
    
    X_train, y_train = fe.choose_features(features_train_df, features, ['longitude', 'latitude']) 
    X_test, y_test = fe.choose_features(features_test_df, features, [])
        
    return X_train, y_train, X_test, y_test


# Helper function: Haversine formula for geodesic distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat/2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Custom loss function: Geodesic distance
def geodesic_loss(y_true, y_pred):
    lat_true, lon_true = y_true[:, 0], y_true[:, 1]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]
    distance = haversine(lat_true, lon_true, lat_pred, lon_pred)
    return distance.mean()

# Custom metric for geodesic distance
def geodesic_metric(y_true, y_pred):
    lat_true, lon_true = y_true[:, 0], y_true[:, 1]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]
    return haversine(lat_true, lon_true, lat_pred, lon_pred).mean()


def train_ensemble_model(model, X_train, y_train): #example of model: MultiOutputRegressor(XGBRegressor(n_estimators=1000, learning_rate=0.1))
    '''
    Train the ensemble model

    Args:
    model: object, the ensemble model to train
    X_train: array, the training features
    y_train: array, the training target

    '''
    # Train the model
    model.fit(X_train, y_train)

    return model

def predict(model, X_test):
    '''
    Make predictions using the ensemble model

    Args:
    model: object, the trained ensemble model
    X_test: array, the test features

    '''

    # Make predictions
    y_pred = model.predict(X_test)

    return y_pred

def evaluate_model(y_pred, y_true):
    '''
    Evaluate the model using the validation set

    Args:
    y_pred: array, the predicted values
    y_true: array, the true values

    '''

    # Prepare output
    y_cat_prep = val.prepare_output(y_pred, y_true)

    # Calculate score
    score = val.calculate_score(y_cat_prep)

    return score


def run_ensemble_model(train_csv, test_csv, vessel_csv, model, features, split_train=False):

    '''
    Run the entire pipeline for the ensemble model

    Args:
    train_csv: str, path to the training data
    test_csv: str, path to the test data
    model: object, the ensemble model to train
    features: list, list of features to use for training the model
    split_train: bool, whether to split the training data into train and validation sets
    evaluate: bool, whether to evaluate the model using the validation set

    '''

    train_df, test_df = process_data(train_csv, test_csv, vessel_csv)

    
    # Prepare the data
    if split_train:

        X_train, y_train, X_val, y_val, validation_df, X_test, y_test = prepare_data_to_train(train_df, test_df, features, split_train=split_train)
         
        # Train the model
        model = train_ensemble_model(model, X_train, y_train)

        # Make predictions
        y_pred = predict(model, X_test)

        y_val_pred = predict(model, X_val)
    
        score = evaluate_model(y_val_pred, validation_df)

        return y_pred, test_df, score
    
    
    X_train, y_train, X_test, y_test = prepare_data_to_train(train_df, test_df, features, split_train=split_train)

    # Train the model
    model = train_ensemble_model(model, X_train, y_train)

    # Make predictions
    y_pred = predict(model, X_test)

    return y_pred, test_df, None

def prepare_kaggle_prediction(sample_submission_csv, predictions_csv, predictions_df, test_df):
    '''
    Prepare the Kaggle submission file

    Args:
    sample_submission_csv: str, path to the sample submission file
    predictions_csv: str, path to save the predictions file
    predictions_df: DataFrame, the predictions
    test_df: DataFrame, the test data
    '''
    model_results = val.prepare_output(predictions_df, test_df)
    ais_sample_submission = pd.read_csv(sample_submission_csv)
    
    # Merge results to the sample submission
    model_submission = pd.merge(ais_sample_submission.drop(columns=['longitude_predicted', 'latitude_predicted']), model_results[['ID', 'longitude_predicted', 'latitude_predicted']], on=['ID'], how='left')

    # Save the submission file
    model_submission.to_csv(predictions_csv, index=False)