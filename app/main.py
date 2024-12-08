from fastapi import FastAPI
from fastapi import FastAPI, Form, File, UploadFile, Request, Depends

from sqlalchemy.orm import Session
from app.routes import get_db

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.routes import router, classify, find_auto_shop_DTC, find_auto_shop_categorie

import os
import json
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")

from huggingface_hub import InferenceClient

categories = ["Auto Repair", "Oil Change Stations", "Transmission Repair", "Smog Check Stations",
              "Tires", "Wheel & Rim Repair", "Body Shops", "DIY Auto Shop",
              "Auto Parts & Supplies", "Hybrid Car Repair", "Car Dealers", "Used Car Dealers",
              "Windshield Installation & Repair", "Towing", "Auto Customization", "Car Inspectors",
              "Auto Detailing", "Commercial Truck Repair", "Gas Stations", "Roadside Assistance",
              "Trailer Repair", "Auto Glass Services", "Interlock Systems", "Machine Shops",
              "Car Buyers", "Auto Upholstery", "Registration Services", "Mobile Dent Repair",
              "Convenience Stores", "Auto Security", "Car Wash", "RV Repair",
              "Truck Rental", "Trailer Rental", "Race Tracks"
            ]

import time
import csv

CITIES_NAMES = ["Los-Angeles", "San-Francisco", "San-Diego", "San-Jose",
          "Fresno", "Sacramento", "Long-Beach", "Oakland",
          "Bakersfield", "Anaheim", "Santa-Ana", "Riverside",
          "Irvine", "Chula-Vista", "Fontana", "Ontario",
          "Modesto", "Glendale", "Huntington Beach", "Lancaster"]

CITIES_LOCATION = {"Los-Angeles": {"lat": 34.0536909, "lon": -118.242766},
                   "San-Francisco": {"lat": 37.7792588, "lon": -122.4193286},
                   "San-Diego": {"lat": 32.7174202, "lon": -117.162772},
                   "San-Jose": {"lat": 37.3361663, "lon": -121.890591},
                   "Fresno": {"lat": 36.7394421, "lon": -119.78483},
                   "Sacramento": {"lat": 38.5810606, "lon": -121.493895},
                   "Long-Beach": {"lat": 33.7690164, "lon": -118.191604},
                   "Oakland": {"lat": 37.8044557, "lon": -122.271356},
                   "Bakersfield": {"lat": 35.3738712, "lon": -119.019463},
                   "Anaheim": {"lat": 33.8347516, "lon": -117.911732},
                   "Santa-Ana": {"lat": 33.7494951, "lon": -117.873221},
                   "Riverside": {"lat": 33.9824949, "lon": -117.374238},
                   "Irvine": {"lat": 33.6856969, "lon": -117.825981},
                   "Chula-Vista": {"lat": 32.6400541, "lon": -117.084195},
                   "Fontana": {"lat": 34.0922947, "lon": -117.43433},
                   "Ontario": {"lat": 34.068241119384766, "lon": -117.65079498291016},
                   "Modesto": {"lat": 37.6393419, "lon": -120.9968892},
                   "Glendale": {"lat": 34.1469416, "lon": -118.2478471},
                   "Huntington-Beach": {"lat": 33.6783336, "lon": -118.000016},
                   "Lancaster": {"lat": 34.6981064, "lon": -118.1366153}}

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
        file_response = await classify(file)

        businesses_response = await find_auto_shop_DTC(city, file_response["Prediciton"], db)

        #TODO: Delete after clasification labeling that do not use multiple error per label
        separated_labels = []
        for item in file_response["Prediciton"]:
            if '-' in item:
                separated_labels.extend(item.split('-'))
            else:
                separated_labels.append(item)

        dict_filename = "obd-trouble-codes.csv"
        trouble_list = []
        with open(dict_filename, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                for resp in separated_labels:
                    if (row[0] == resp):
                        trouble_list.append(str(row[1]) + "(" + str(file_response["Percentages"][resp]) + "%)")


        locations = []
        for biz in businesses_response:
            locations.append({"name": biz["name"], "lat": biz["lat"], "lon": biz["lon"]})

        return templates.TemplateResponse("index.html", {
            "request": request,
            "cities": CITIES_NAMES,
            "troubles": trouble_list,
            "businesses": businesses_response,
            "city": city,
            "map_loc": CITIES_LOCATION[city]
        })
    elif submit_button == "Dialog_button":        
        client = InferenceClient(
            "microsoft/Phi-3-mini-4k-instruct",
            token=api_key,
        )

        #My car makes a loud noise when braking.

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

        start_idx = resp.find("RESPONSE= ")
        response_text = ""
        if start_idx != -1:
            start_idx += len("RESPONSE= ")
            end_idx = resp.find("\n", start_idx)
            if end_idx == -1:
                response_text = resp[start_idx:]
            response_text = resp[start_idx:end_idx]
        
        start_idx = resp.find("QUERY_CATEGORIES= ")
        query_cat = ""
        if start_idx != -1:
            start_idx += len("QUERY_CATEGORIES= ")
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

    #return templates.TemplateResponse("index.html", {"request": request, "time": int(time.time()), "cities": CITIES_NAMES})
