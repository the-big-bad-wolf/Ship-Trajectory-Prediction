import pandas as pd
import numpy as np
import feature_engineering as fe
import validation as val

def prepare_data(train_csv, test_csv, features, split_train=False):
    '''
    Prepare the data for training and testing the model

    Args:
    training: str, path to the training data
    test: str, path to the test data
    features: list, list of features to use for training the model
    split_train: bool, whether to split the training data into train and validation sets

    '''
    # Load the data
    ais_train_df = pd.read_csv(train_csv, sep='|')
    ais_test_df = pd.read_csv(test_csv, sep=',')

    # Assuming 'vesselId' is the categorical column with 688 unique entries
    ais_train_df, ais_test_df = fe.transform_categorical(ais_train_df, ais_test_df, 'vesselId')

    # Feature engineering
    features_train_df = fe.feature_engineer_train(ais_train_df)
    features_test_df = fe.feature_engineer_test(ais_train_df, ais_test_df)

    # Split the data into train and validation sets
    if split_train:
        train_df, validation_df = fe.split_train_validation(features_train_df)
        X_train, y_train = fe.choose_features(train_df, features, ['longitude', 'latitude']) 
        X_val, y_val = fe.choose_features(validation_df, features, ['longitude', 'latitude'])
        X_test, y_test = fe.choose_features(features_test_df, features, [])
        
        return X_train, y_train, X_val, y_val, validation_df, X_test, y_test
    
    X_train, y_train = fe.choose_features(train_df, features, ['longitude', 'latitude']) 
    X_test, y_test = fe.choose_features(features_test_df, features, [])
    
    return X_train, y_train, X_test, y_test


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


def run_ensemble_model(train_csv, test_csv, model, features, split_train=False, evaluate=False):
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
    # Prepare the data
    if split_train:
        X_train, y_train, X_val, y_val, validation_df, X_test, y_test = prepare_data(train_csv, test_csv, features, split_train=split_train)
    else:
        X_train, y_train, X_test, y_test = prepare_data(train_csv, train_csv, features, split_train=split_train)

    # Train the model
    model = train_ensemble_model(model, X_train, y_train)

    # Make predictions
    y_pred = predict(model, X_test)

    # Evaluate the model
    if evaluate:
        y_val_pred = predict(model, X_val)
        y = val.prepare_output(y_val_pred, validation_df)
        score = evaluate_model(y)
        return y_pred, score

    return y_pred, None