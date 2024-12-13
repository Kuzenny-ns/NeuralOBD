from fastapi import APIRouter, Depends, UploadFile, File
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from app.models import AutoShop, Category, Location, Error
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
async def classify(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # model = tf.keras.models.load_model('models/trouble_code_classifier_tensorflow_100_epochs_acc95.keras')
    model = tf.keras.models.load_model('models/trouble_code_classifier_sigmoid_output.keras')
    
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
    
    threshold = 0.99
    predicted_classes = (predictions > threshold).astype(int)
    
    error_names = db.query(Error.error_code).all()
    error_classes = [error[0] for error in error_names]

    predicted_class_names = [[error_names[i] for i, val in enumerate(row) if val == 1] for row in predicted_classes]

    flattened_classes = [item for sublist in predicted_class_names for item in sublist]
    class_counts = Counter(flattened_classes)
    total_classes = len(flattened_classes)
    class_percentages = {class_label: (count / total_classes) * 100 for class_label, count in class_counts.items()}

    print("\n\n")
    for class_label, percentage in class_percentages.items():
        print(f"{class_label[0]}: {percentage:.2f}%")
    print("\n\n")

    return {"Posible errors": error_classes,
            "Prediciton": class_percentages}

@router.get("/findAutoShopDTC/")
async def find_auto_shop_DTC(city: str, trouble_code: List[str], db: Session = Depends(get_db)):
    errors_with_categories = db.query(Error).all()
    trouble_dict = {
        error.error_code: [category.name for category in error.categories]
        for error in errors_with_categories
    }

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