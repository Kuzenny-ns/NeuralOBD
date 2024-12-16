import pandas as pd
from sqlalchemy import create_engine

from app.models import AutoShop, Category, Location, Error
from app.models import SessionLocal
import os
from dotenv import load_dotenv
load_dotenv()
username = os.getenv("DB_username")
password = os.getenv("DB_password")
host = os.getenv("DB_host")
database = os.getenv("DB_database")

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

def create_DB_from_csv():
    # csv_file = "cleaned_file.csv"
    csv_file = "SCRAPING_CLEANED_DROPNA.csv"
    df = pd.read_csv(csv_file)

    engine = create_engine(url=f"mysql+mysqlconnector://{username}:{password}@{host}/{database}")

    table_name = "auto_repair_shops"
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    print(f"Data has been successfully loaded into the '{table_name}' table in the '{database}' database!")

def populate():
    db = SessionLocal()

    # df = pd.read_csv("cleaned_file.csv")
    df = pd.read_csv("SCRAPING_CLEANED_DROPNA.csv")
    df = df.where(pd.notnull(df), None)

    df["categories"] = df["categories"].apply(lambda x: eval(x))

    category_cache = {}
    location_cache = {}

    for _, row in df.iterrows():
        #Check if shop already exists
        existing_shop = db.query(AutoShop).filter_by(biz_id=row["bizId"]).first()

        
        location_key = (row["city"], row["region"])
        if location_key not in location_cache:
            coordinates = CITIES_LOCATION.get(row["city"])
            location = Location(city=row["city"], region=row["region"], latitude=coordinates.get("lat"), longitude=coordinates.get("lon"))
            db.add(location)
            db.commit()
            location_cache[location_key] = location
        else:
            location = location_cache[location_key]

        if existing_shop:
            if location not in existing_shop.locations:
                existing_shop.locations.append(location)
        else:
            shop = AutoShop(
                biz_id=row["bizId"],
                name=row["name"],
                rating=row["rating"],
                review_count=row["reviewCount"],
                formatted_address=row["formattedAddress"],
                latitude = row["latitude"],
                longitude = row["longitude"],
                is_ad=row["isAd"],
                photo_url=row["photoUrl"],
                business_url=row["businessUrl"],
            )
            shop.locations.append(location)
            db.add(shop)
            db.commit()

            for category_name in row["categories"]:
                if category_name not in category_cache:
                    category = Category(name=category_name)
                    db.add(category)
                    db.commit()
                    category_cache[category_name] = category
                else:
                    category = category_cache[category_name]

                shop.categories.append(category)

            db.commit()

    df2 = pd.read_csv("obd-trouble-codes-for-db.csv")
    df2["categories"] = df2["categories"].apply(lambda x: eval(x))

    for index, row in df2.iterrows():
        error_text = row['error']
        description = row['description']

        error = Error(error_code=error_text, description=description)
        db.add(error)
        db.commit()

        for category_name in row["categories"]:
            if category_name in category_cache:
                category = category_cache[category_name]
                error.categories.append(category)
        
        db.commit()


    print("Data has been successfully loaded into the database!")
