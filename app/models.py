from sqlalchemy import Column, Integer, String, Boolean, Float, Double, ForeignKey, Table, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker

import os
from dotenv import load_dotenv
load_dotenv()
username = os.getenv("DB_username")
password = os.getenv("DB_password")
host = os.getenv("DB_host")
database = os.getenv("DB_database")

Base = declarative_base()

shop_categories = Table(
    "shop_categories",
    Base.metadata,
    Column("shop_id", Integer, ForeignKey("auto_shops.id"), primary_key=True),
    Column("category_id", Integer, ForeignKey("categories.id"), primary_key=True),
)

shop_locations = Table(
    "shop_locations",
    Base.metadata,
    Column("shop_id", Integer, ForeignKey("auto_shops.id"), primary_key=True),
    Column("location_id", Integer, ForeignKey("locations.id"), primary_key=True),
)

class AutoShop(Base):
    __tablename__ = "auto_shops"

    id = Column(Integer, primary_key=True, index=True)
    biz_id = Column(String(255), unique=True, index=True)
    name = Column(String(255), nullable=False)
    rating = Column(Float)
    review_count = Column(Integer)
    formatted_address = Column(String(255))
    latitude = Column(Double)
    longitude = Column(Double)
    is_ad = Column(Boolean)
    photo_url = Column(String(255))
    business_url = Column(String(255))

    categories = relationship("Category", secondary=shop_categories, back_populates="shops")
    locations = relationship("Location", secondary=shop_locations, back_populates="shops")


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)

    shops = relationship("AutoShop", secondary=shop_categories, back_populates="categories")

class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    city = Column(String(255), nullable=False)
    region = Column(String(255), nullable=False)

    shops = relationship("AutoShop", secondary=shop_locations, back_populates="locations")


engine = create_engine(url=f"mysql+mysqlconnector://{username}:{password}@{host}/{database}")
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
