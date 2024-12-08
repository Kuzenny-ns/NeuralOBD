import pandas as pd
from sqlalchemy import create_engine

from app.models import AutoShop, Category, Location
from app.models import SessionLocal
import os
from dotenv import load_dotenv
load_dotenv()
username = os.getenv("DB_username")
password = os.getenv("DB_password")
host = os.getenv("DB_host")
database = os.getenv("DB_database")

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
            location = Location(city=row["city"], region=row["region"])
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

    print("Data has been successfully loaded into the database!")


#create_DB_from_csv()
#populate()