import pandas as pd
import numpy as np
import re
import os
import sys
import csv

# 1. setup: defining paths
print("Phase 1 (Lightweight): Text-Only Data Preparation - STARTING")
print(f"Current working directory: {os.getcwd()}")

RAW_DATA_FOLDER = 'data/raw'
PROCESSED_DATA_FOLDER = 'data/processed'

TRAIN_FILE_PATH = os.path.join(RAW_DATA_FOLDER, 'train.csv')
TEST_FILE_PATH = os.path.join(RAW_DATA_FOLDER, 'test.csv')

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)
print("Project directories verified.")

# 2. load raw data
print("Loading raw data from {RAW_DATA_FOLDER}...")

try:
    # set a large field size limit for safety
    max_int = sys.maxsize
    decrement = True
    while decrement:
        try:
            csv.field_size_limit(max_int)
            decrement = False
        except OverflowError:
            max_int = int(max_int / 10)
    print("CSV field size limit set.")

    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)

    print(f"Successfully loaded train.csv. Total rows: {len(train_df)}")
    print(f"Successfully loaded test.csv. Total rows: {len(test_df)}")

    # Verify the data is complete
    if len(train_df) < 75000:
        print("\n\n*** CRITICAL WARNING ***")
        print(f"Your train.csv file only has {len(train_df)} rows.")
        print("It is incomplete. You MUST re-download the full 75,000-row file.")
        print("Stopping script.")
        sys.exit()

except Exception as e:
    print(f"\n\n*** CRITICAL ERROR LOADING CSV ***")
    print(f"Error: {e}")
    print("This likely means your .csv files are corrupt or incomplete.")
    print(f"Please place your full CSV files in '{RAW_DATA_FOLDER}' and run again.")
    sys.exit()

# 3. feature extraction from 'catalog_content'
print("\nStarting feature extraction from 'catalog_content'...")

# Regex to find pack size, e.g., "(Pack of 6)", "12 per case", "Count 12"
pack_regex = re.compile(r'(?:pack of|count|per case|pack)\s*\(?(\d+)\)?',
                        re.IGNORECASE
)

# Regex to find weight/volume, e.g., "11.25 oz", "1.9 Ounce", "12.7 Fl Oz"
value_regex = re.compile(r'value:\s*([\d\.]+)', re.IGNORECASE)
unit_regex = re.compile(r'unit:\s*([\w\s]+)', re.IGNORECASE)

def extract_features(text_series):
    text_series = text_series.astype(str)
    pack_sizes = text_series.str.extract(pack_regex, expand=False)\
                            .fillna(1).astype(float)
    values = text_series.str.extract(value_regex, expand=False)\
                        .fillna(0).astype(float)
    units = text_series.str.extract(unit_regex, expand=False)\
                       .fillna('Unknown').str.strip().str.lower()
    return pack_sizes, values, units

def create_feature_df(df, median_measure=None, median_pack_size=None):
    pack_sizes, values, units = extract_features(df['catalog_content'])

    feature_df = pd.DataFrame({
        'pack_size': pack_sizes,
        'total_measure': values,
        'unit': units
    })

    unit_dummies = pd.get_dummies(feature_df['unit'], prefix='unit', dummy_na=False)
    features_df = pd.concat([feature_df, unit_dummies], axis=1)
    features_df = features_df.drop('unit', axis=1)

    # if this is not training set, calculate medians
    if median_measure is None:
        median_measure = features_df['total_measure'].median()
    if median_pack_size is None:
        median_pack_size = features_df['pack_size'].median()
    
    #fill any missing values with medians
    features_df['total_measure'] = features_df['total_measure'].fillna(median_measure)
    features_df['pack_size'] = features_df['pack_size'].fillna(median_pack_size)

    return features_df, median_measure, median_pack_size

# applying to both train and test data
print("\nProcessing training data...")
train_features_df, median_measure, median_pack_size = create_feature_df(train_df)

print("Processing test features...")
# Use the medians from the *training* data to fill missing values in the test set
test_features_df, _, _ = create_feature_df(
    test_df, 
    median_measure=median_measure, 
    median_pack_size=median_pack_size
)

# align columns
print("Aligning train/test feature columns...")
train_cols, test_cols = train_features_df.align(test_features_df, join='inner', axis=1, fill_value=0)

print("Feature extraction complete.")
print(f"Shape of extracted train features: {train_cols.shape}")
print(f"Shape of extracted test features: {test_cols.shape}")

# 4. save processed data
print(f"\nSaving processed data to '{PROCESSED_DATA_FOLDER}'...")

# Combine original data with new features (not using image_link for now)
train_df = train_df.drop('image_link', axis=1)
test_df = test_df.drop('image_link', axis=1)

train_final = pd.concat([train_df, train_cols], axis=1)
test_final = pd.concat([test_df, test_cols], axis=1)

# add log_price target variable
train_final['log_price'] = np.log1p(train_final['price'])

# Using .parquet is much faster for loading/saving later
train_final.to_parquet(os.path.join(PROCESSED_DATA_FOLDER, 'train_processed.parquet'))
test_final.to_parquet(os.path.join(PROCESSED_DATA_FOLDER, 'test_processed.parquet'))

print("\n--- PHASE 1 (Lightweight) SCRIPT COMPLETE ---")
print(f"Processed dataframes (text-only) are in: {PROCESSED_DATA_FOLDER}")