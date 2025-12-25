from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, coo_matrix

app = FastAPI()

model = joblib.load('models/champion_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
brand_encoder = joblib.load('models/brand_encoder.pkl')
num_features = joblib.load('models/numerical_features.pkl')

class ProductInput(BaseModel):
    description: str
    pack_size: float
    total_measure: float

@app.post("/predict")
def predict_price(data: ProductInput):
    text_vec = tfidf.transform([data.description])

    brand_regex = re.compile(r'Item Name: \s*([\w\’\'\-\.&]+)')
    match = brand_regex.search(data.description)
    brand = match.group(1).lower() if match else 'unknown'

    try:
        brand_encoded = brand_encoder.transform([brand])[0]
    except:
        brand_encoded = brand_encoder.transform(['unknown'])[0]

    num_data = pd.DataFrame(0, index=[0], columns=num_features)
    num_data['pack_size'] = data.pack_size
    num_data['total_measure'] = data.total_measure

    X_num = num_data.values.astype(np.float32)
    X_brand = np.array([[brand_encoded]]).astype(np.float32)

    X_final = hstack((text_vec, coo_matrix(X_num), coo_matrix(X_brand)))

    log_pred = model.predict(X_final)
    price = float(np.expm1(log_pred)[0])

    return {"estimated_price": round(price, 2), "brand_detected": brand}