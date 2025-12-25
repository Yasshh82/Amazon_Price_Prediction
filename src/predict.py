import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack, coo_matrix

print("loading models and encoders...")
model = joblib.load('models/champion_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
brand_encoder = joblib.load('models/brand_encoder.pkl')
num_features_list = joblib.load('models/numerical_features.pkl')

def get_prediction(catalog_content, pack_size, total_measure):
    text_data = tfidf.transform([catalog_content])

    brand_regex = re.compile(r'Item Name:\s*([\w\’\'\-\.&]+)')
    match = brand_regex.search(catalog_content)
    brand = match.group(1).lower() if match else 'unknown'

    try:
        brand_encoded = brand_encoder.transform([brand])[0]
    except ValueError:
        brand_encoded = brand_encoder.transform(['unknown'])[0]

    num_data = pd.DataFrame(0, index=[0], columns=num_features_list)
    num_data['pack_size'] = float(pack_size)
    num_data['total_measure'] = float(total_measure)

    X_num = num_data.values.astype(np.float32)
    X_brand = np.array([[brand_encoded]]).astype(np.float32)

    X_final = hstack((
        text_data,
        coo_matrix(X_num),
        coo_matrix(X_brand)
    ))

    log_pred = model.predict(X_final)
    price = np.expm1(log_pred)[0]

    return price

print("\n---Price Predictor---")
print("Enter details to estimate the price (Type 'exit' to stop)")

while True:
    desc = input("\nProduct Description (e.g., Item Name: La Victoria Sauce..): ")
    if desc.lower() == 'exit': break

    ps = input("Pack Size (number): ")
    tm = input("Total Measure (number): ")

    try:
        est_price = get_prediction(desc, ps, tm)
        print(f"\n>>> Estimated Price: ${est_price:.2f}")
    except Exception as e:
        print(f"Error: {e}")