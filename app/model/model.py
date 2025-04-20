import torch
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import requests
import logging
import traceback
import io

from app.model.TimeSeriesDataset import TimeSeriesDataset
from app.model.implementation import TCNForecaster

logger = logging.getLogger(__name__)

TARGET_COLUMN = "wl-a"
SEQUENCE_LENGTH = 6

# Raw GitHub URLs to your model + scaler
MODEL_URL = "https://drive.google.com/uc?export=download&id=1JtYnxXkQzCQG89jMmxDwycDXzwBD6OgV"
SCALER_URL = "https://drive.google.com/uc?export=download&id=1qR5EgoO1Um-SFH2WA_gdysU60zI3wbJx"

model = None
scaler = None

# These must match what was used during training
base_features = ["rf-a", "rf-a-sum", "wl-ch-a"]
temporal_features = ["day_of_week", "week", "month", "year"]
rolling_windows = {'mean': [1, 2, 3]}
lags = [1, 2, 4]


def load_model():
    """Load model and scaler directly from GitHub URLs"""
    global model, scaler

    try:
        # Load model directly from GitHub URL
        model = TCNForecaster(input_size=len(get_all_feature_names()), output_size=1, num_channels=[62, 128, 256])

        # Get model file from GitHub
        model_response = requests.get(MODEL_URL)
        model_response.raise_for_status()

        # Load model state dict from the response content
        model_buffer = io.BytesIO(model_response.content)
        state_dict = torch.load(model_buffer, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()

        # Get scaler file from GitHub
        scaler_response = requests.get(SCALER_URL)
        scaler_response.raise_for_status()

        # Load scaler from the response content
        scaler_buffer = io.BytesIO(scaler_response.content)
        scaler = joblib.load(scaler_buffer)

        logger.info("Model and scaler loaded successfully from GitHub.")
        print("[INFO] Model and scaler loaded successfully from GitHub.")
        return True

    except Exception as e:
        logger.error(f"Failed to load model or scaler: {e}")
        logger.error(traceback.format_exc())
        print(f"[ERROR] Failed to load model or scaler: {e}")
        print(traceback.format_exc())
        return False

def preprocess_data(df):
    # Match the preprocessing steps from training code
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.sort_values('Datetime', inplace=True)

    # Target is shifted in the training code
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].shift(-1)  # Shift to predict next value

    # Check for missing base features
    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        logger.warning(f"Adding missing base features with zeros: {missing_features}")
        for f in missing_features:
            df[f] = 0.0

    # Clean up the data - exact same steps as in training
    df = df.replace(r'\(\*\)', np.nan, regex=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.interpolate(method='linear', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def add_temporal_features(df):
    # Ensure this exactly matches the training code's time_temporal_features_extraction function
    if 'Datetime' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Datetime']):
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

        df['day_of_week'] = df['Datetime'].dt.dayofweek  # Changed from dayofweek.astype(int)
        df['week'] = df['Datetime'].dt.isocalendar().week  # Changed from week.astype(int)
        df['month'] = df['Datetime'].dt.month  # Changed from month.astype(int)
        df['year'] = df['Datetime'].dt.year  # Changed from year.astype(int)
    else:
        logger.error("Datetime column not found in the dataframe")
        df['day_of_week'] = 0
        df['week'] = 1
        df['month'] = 1
        df['year'] = 2025

    return df


def add_rolling_and_lagged_features(df):
    # Match the rolling_features function from training code
    for feature in base_features:
        if feature in df.columns:
            for window_size in rolling_windows['mean']:
                df[f'{feature}_{window_size}0min_avg'] = df[feature].rolling(window=window_size, min_periods=1).mean()

    # Add lagged features
    for feature in base_features:
        if feature in df.columns:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def get_all_feature_names():
    # Exactly match the feature selection from training code
    feature_names = base_features.copy()

    # Add temporal features
    feature_names.extend(temporal_features)

    # Add rolling window features
    for feature in base_features:
        for window_size in rolling_windows['mean']:
            feature_names.append(f'{feature}_{window_size}0min_avg')

    # Add lagged features
    for feature in base_features:
        for lag in lags:
            feature_names.append(f'{feature}_lag_{lag}')

    logger.info(f"Total feature count: {len(feature_names)}, Features: {feature_names}")
    return feature_names


def predict_pipeline(df):
    global model, scaler

    if model is None or scaler is None:
        if not load_model():
            return {"error": "Model or scaler could not be loaded"}

    try:
        # Store the last datetime for future prediction timestamp
        df = preprocess_data(df)
        last_datetime = pd.to_datetime(df['Datetime'].iloc[-1])

        # Apply feature engineering in the same order as training
        df = add_temporal_features(df)
        df = add_rolling_and_lagged_features(df)

        # Get all feature names in correct order
        all_features = get_all_feature_names()

        # Check for missing features
        missing_features = [f for f in all_features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return {"error": f"Missing required features: {missing_features}"}

        logger.info(f"Feature dimensions before scaling: {df[all_features].shape}")

        # Important: Only scale the numerical features that were in the training data
        # Mirroring exactly how the scaler was used in training
        numerical_cols = base_features  # Just the base numerical features
        numerical_data = df[numerical_cols]

        # Apply scaling only to numerical columns
        numerical_scaled = scaler.transform(numerical_data)

        # Create a new DataFrame with scaled numerical features
        scaled_df = df.copy()
        for i, col in enumerate(numerical_cols):
            scaled_df[col] = numerical_scaled[:, i]

        # Extract features in the correct order for the model
        features = np.array(scaled_df[all_features]).astype(np.float32)

        logger.info(f"Feature dimensions after processing: {features.shape}")

        if len(features) < SEQUENCE_LENGTH:
            return {"error": f"Insufficient data. Require {SEQUENCE_LENGTH}, got {len(features)}"}

        # Create tensor dataset
        features_tensor = torch.FloatTensor(features)
        dataset = TimeSeriesDataset(features=features_tensor, targets=None, sequence_length=SEQUENCE_LENGTH)
        dataloader = DataLoader(dataset, batch_size=24, shuffle=False)

        # Log shape for debugging
        for batch_x in dataloader:
            logger.info(f"TCN input tensor shape: {batch_x.shape}")
            break

        # Make predictions
        predictions = []
        with torch.no_grad():
            for batch_x in dataloader:
                output = model(batch_x)
                predictions.extend(output.numpy().flatten())

        # Get the last prediction (10 minutes in the future)
        predicted_value = predictions[-1] if predictions else None
        future_time = last_datetime + timedelta(minutes=10)

        return {
            "datetime": future_time.strftime('%Y-%m-%d %H:%M:%S'),
            "wl-a_predicted": round(float(predicted_value), 4) if predicted_value is not None else None,
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return {"error": str(e), "traceback": traceback.format_exc()}
