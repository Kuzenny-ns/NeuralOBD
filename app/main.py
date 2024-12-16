from fastapi import FastAPI
from fastapi import FastAPI, Form, File, UploadFile, Request, Depends

from sqlalchemy.orm import Session
from app.routes import get_db

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.routes import router, classify, find_auto_shop_DTC, find_auto_shop_categorie
from app.models import Category, Location, Error, SessionLocal

import time

import os
import json
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")

from huggingface_hub import InferenceClient

session = SessionLocal()
cats = session.query(Category.name).all()
categories = [name[0] for name in cats]

city_names = session.query(Location.city).all()
CITIES_NAMES = [city[0] for city in city_names]

cities_data = session.query(Location).all()
CITIES_LOCATION = {
    city.city: {"lat": city.latitude, "lon": city.longitude}
    for city in cities_data
}
session.close()

templates = Jinja2Templates(directory="app\\templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="app\static"), name="static")

app.include_router(router)

@app.get("/")
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "time": int(time.time()), "cities": CITIES_NAMES})

@app.post("/")
async def post_root(request: Request, file: UploadFile = File(...),
                    city: str = Form(...), submit_button: str = Form(...),
                    dialog_input: str = Form(...), db: Session = Depends(get_db)):
    
    if submit_button == "CSV_button":
        file_response = await classify(file, db)

        trouble_dict = {}
        for class_label, percentage in file_response["Prediciton"].items():
            trouble_dict[class_label[0]] = percentage

        trouble_list = []
        if len(trouble_dict) > 1:
            del trouble_dict["NO_ERROR"]

        for key in list(trouble_dict.keys()):
            if trouble_dict[key] < 4.5:
                del trouble_dict[key]

        for class_label, percentage in trouble_dict.items():
            trouble_list.append(class_label)
            print(f"{class_label}: {percentage:.2f}%")

        results = db.query(Error).filter(Error.error_code.in_(trouble_list)).all()
        descriptions = [result.description for result in results]

        businesses_response = await find_auto_shop_DTC(city, trouble_list, db)

        locations = []
        for biz in businesses_response:
            locations.append({"name": biz["name"], "lat": biz["lat"], "lon": biz["lon"]})

        return templates.TemplateResponse("index.html", {
            "request": request,
            "cities": CITIES_NAMES,
            "troubles": descriptions,
            "businesses": businesses_response,
            "city": city,
            "map_loc": CITIES_LOCATION[city]
        })
    elif submit_button == "Dialog_button":        
        client = InferenceClient(
            "microsoft/Phi-3-mini-4k-instruct",
            token=api_key,
        )

        prompt = f"User's description of car problem: '{dialog_input}'\n\n" \
                f"Possible categories of auto shop services:\n" + "\n".join([f"- {cat}" for cat in categories]) + "\n\n" \
                "Based on the user's description, generate a short response and give relevant categories of auto shop services." \
                "Put categories list in square brackets and write 'QUERY_CATEGORIES=' before the list. Before response write 'RESPONSE=' and put response text in square brackets." \
                "In response only talk about what is wrong with car."
        
        response = client.post(
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 200},
                "task": "text-generation",
            },
        )
        resp = json.loads(response.decode())[0]["generated_text"]
        print(resp)

        start_idx = resp.find("RESPONSE=[")
        response_text = ""
        if start_idx != -1:
            start_idx += len("RESPONSE=")
            end_idx = resp.find("\n", start_idx)
            if end_idx == -1:
                response_text = resp[start_idx:]
            response_text = resp[start_idx:end_idx]
        
        start_idx = resp.find("QUERY_CATEGORIES=[")
        query_cat = ""
        if start_idx != -1:
            start_idx += len("QUERY_CATEGORIES=")
            end_idx = resp.find("\n", start_idx)
            if end_idx == -1:
                query_cat = resp[start_idx:]
            query_cat = resp[start_idx:end_idx]

        response_text_list = []
        response_text_list.append(response_text.replace('[', '').replace(']', ''))
        query_cat_list = query_cat.replace('[', '').replace(']', '').split(', ')
        print(f"\n\n\nResponse ={response_text}\nCategories ={query_cat_list}\n\n\n")
        businesses_response = await find_auto_shop_categorie(city, query_cat_list, db)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "cities": CITIES_NAMES,
            "troubles": response_text_list,
            "businesses": businesses_response,
            "city": city,
            "map_loc": CITIES_LOCATION[city]
        })
