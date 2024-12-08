from fastapi import APIRouter, Depends, UploadFile, File
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from app.models import AutoShop, Category, Location
from app.models import SessionLocal
from io import StringIO
from typing import List

from models.model_training import normalize_percentages_and_formating


from collections import Counter
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter()

@router.post("/diagnose/")
async def diagnose():
    return {"message": "Diagnosis endpoint"}

@router.get("/classify/")
async def classify(file: UploadFile = File(...)):
    model = tf.keras.models.load_model('models/trouble_code_classifier_tensorflow_100_epochs_acc95.keras')
    
    content = await file.read()
    csv_data = StringIO(content.decode("utf-8"))
    df = pd.read_csv(csv_data)

    df['ENGINE_LOAD'] = normalize_percentages_and_formating(df['ENGINE_LOAD'])
    df['THROTTLE_POS'] = normalize_percentages_and_formating(df['THROTTLE_POS'])
    df['TIMING_ADVANCE'] = normalize_percentages_and_formating(df['TIMING_ADVANCE'])
    df['ENGINE_POWER'] = normalize_percentages_and_formating(df['ENGINE_POWER'])
    
    feature_columns = ['ENGINE_POWER', 'ENGINE_COOLANT_TEMP', 'ENGINE_LOAD', 'ENGINE_RPM','INTAKE_MANIFOLD_PRESSURE',
                       'AIR_INTAKE_TEMP', 'THROTTLE_POS', 'TIMING_ADVANCE']
    X = df[feature_columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    predictions = model.predict(X_scaled)
    
    predicted_classes = predictions.argmax(axis=-1)
    
    error_classes = ['C0300', 'NO_ERROR', 'P0078-B0004-P3000', 'P0078-U1004-P3000',
                     'P0079-C1004-P3000', 'P0079-P1004-P3000', 'P0079-P2004-P3000', 'P007E-P2036-P18D0',
                     'P007E-P2036-P18E0', 'P007E-P2036-P18F0', 'P007F-P2036-P18D0', 'P007F-P2036-P18E0',
                     'P007F-P2036-P18F0', 'P0133']
    predicted_labels = [error_classes[class_idx] for class_idx in predicted_classes]

    counts = Counter(predicted_labels)
    percentage_dict = {}
    total = len(predicted_labels)
    for item, count in counts.items():
        percentage_dict[item] = round((count / total) * 100, 2)
        print(f"{item} = {count}")


    return {"Posible errors": error_classes,
            "Percentages": percentage_dict,
            "Prediciton": list(dict.fromkeys(predicted_labels))}

@router.get("/findAutoShopDTC/")
async def find_auto_shop_DTC(city: str, trouble_code: List[str], db: Session = Depends(get_db)):
    trouble_dict = {"C0300": ['Auto Repair', 'DIY Auto Shop', 'Transmission Repair'],
                    'NO_ERROR': ['Auto Repair'],
                    'P0078-B0004-P3000': ['Auto Repair', 'Smog Check Stations', 'Wheel & Rim Repair'],
                    'P0078-U1004-P3000': ['Auto Repair', 'Smog Check Stations'],
                    'P0079-C1004-P3000': ['Auto Repair', 'Smog Check Stations', 'Wheel & Rim Repair'],
                    'P0079-P1004-P3000': ['Auto Repair', 'Smog Check Stations'],
                    'P0079-P2004-P3000': ['Auto Repair', 'Smog Check Stations'],
                    'P007E-P2036-P18D0': ['Auto Repair'],
                    'P007E-P2036-P18E0': ['Auto Repair'],
                    'P007E-P2036-P18F0': ['Auto Repair'],
                    'P007F-P2036-P18D0': ['Auto Repair'],
                    'P007F-P2036-P18E0': ['Auto Repair'],
                    'P007F-P2036-P18F0': ['Auto Repair'],
                    'P0133': ['Auto Repair']}
    
    category_list = []
    for code in trouble_code:
        category_list.extend(trouble_dict[code])
    category_list = list(dict.fromkeys(category_list))


    #Query to find businesses based on city and having appropriate category
    results = db.query(AutoShop).join(AutoShop.locations).join(AutoShop.categories).filter(
        Location.city == city,
        Category.name.in_(category_list)
    ).order_by(AutoShop.rating.desc()).all()

    response = []
    for carshop in results:
        categories = [category.name for category in carshop.categories]
        response.append({
            "id": carshop.id,
            "biz_id": carshop.biz_id,
            "name": carshop.name,
            "rating": carshop.rating,
            "address": carshop.formatted_address,
            "lat": carshop.latitude,
            "lon": carshop.longitude,
            "city": city,
            "categories": categories,
            "photo_link": carshop.photo_url,
            "link": carshop.business_url
        })
    
    return response

@router.get("/findAutoShopCategires/")
async def find_auto_shop_categorie(city: str, category_list: List[str], db: Session = Depends(get_db)):
    results = db.query(AutoShop).join(AutoShop.locations).join(AutoShop.categories).filter(
        Location.city == city,
        Category.name.in_(category_list)
    ).order_by(AutoShop.rating.desc()).all()

    response = []
    for carshop in results:
        categories = [category.name for category in carshop.categories]
        response.append({
            "id": carshop.id,
            "biz_id": carshop.biz_id,
            "name": carshop.name,
            "rating": carshop.rating,
            "address": carshop.formatted_address,
            "lat": carshop.latitude,
            "lon": carshop.longitude,
            "city": city,
            "categories": categories,
            "photo_link": carshop.photo_url,
            "link": carshop.business_url
        })
    
    return response