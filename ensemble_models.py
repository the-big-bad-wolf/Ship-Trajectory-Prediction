import pandas as pd
import numpy as np
import feature_engineering as fe
import validation as val

def prepare_data(training, test, features, split_train=False):
    # Load the data
    ais_train_df = pd.read_csv('ais_train.csv', sep='|')
    ais_test_df = pd.read_csv('ais_test.csv', sep=',')

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
    # Train the model
    model.fit(X_train, y_train)

    return model

def predict(model, X_test):

    # Make predictions
    y_pred = model.predict(X_test)

    return y_pred

def evaluate_model(y_pred, y_true):
    # Prepare output
    y_cat_prep = val.prepare_output(y_pred, y_true)

    # Calculate score
    score = val.calculate_score(y_cat_prep)

    return score


def run_ensemble_model(train_csv, test_csv, model, features, split_train=False, evaluate=False):
    # Prepare the data
    if split_train:
        X_train, y_train, X_val, y_val, validation_df, X_test, y_test = prepare_data('ais_train.csv', 'ais_test.csv', features, split_train=split_train)
    else:
        X_train, y_train, X_test, y_test = prepare_data('ais_train.csv', 'ais_test.csv', features, split_train=split_train)

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