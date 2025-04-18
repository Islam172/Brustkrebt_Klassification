import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
import fastapi
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

import pandas as pd
import pymongo

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.pipeline.trainig_pipeline import TrainingPipeline
from src.utils.utils import load_object
from src.utils.predictor import Model

# MongoDB setup
mongo_db_url = os.getenv("MONGODB_URL_KEY")
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from src.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from src.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# FastAPI setup
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["Welcome"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        raise CustomException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if "diagnosis" in df.columns:
           df = df.drop(columns=["diagnosis"])
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        model = Model(preprocessor=preprocessor, model=final_model)
        y_pred = model.predict(df)

        df['prediction'] = y_pred

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv")

        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
