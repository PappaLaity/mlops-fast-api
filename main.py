import asyncio
import json
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from src.entities.iris import IrisData
from src.functions import load_model

load_dotenv()
models = {}
classes = ["setosa", "versicolor", "virginica"]
delay = 5
delay2 = 10
LOGISTIC_REGRESSION = os.getenv("LR")
RANDOM_FOREST = os.getenv("RF")
PREDICTIONS_FILE_PATH = os.getenv("Pred_File_Path")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    print("Loading the machine learning model...")
    models["logistic_regression"] = load_model(LOGISTIC_REGRESSION)
    models["random_forest"] = load_model(RANDOM_FOREST)
    yield
    # Shutdown: Unload or clean up resources
    # print("Unloading the machine learning model...")
    models.clear()  # Clear the model from memory


app = FastAPI(lifespan=lifespan)


# Health check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


async def write_prediction_into_file(raw_data):
    await asyncio.sleep(10)
    try:
        with open(PREDICTIONS_FILE_PATH, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        data = []
    except json.JSONDecodeError:
        # If the file is not valid JSON, handle the error or start fresh
        print("Error: Invalid JSON in the file. Starting with an empty list.")
        data = []

    if isinstance(data, list):
        data.append(raw_data)
    else:
        print("Warning: JSON root is not a list. Cannot append directly.")
    # 3. Write the updated JSON data back to the file
    with open(PREDICTIONS_FILE_PATH, "w") as f:
        json.dump(data, f, indent=4)  # Use indent for pretty-printing

    print(f"Row added to {PREDICTIONS_FILE_PATH} successfully.")


@app.post("/predict/logistic_regression")
async def predict_by_logistic_regression(
    data: IrisData, backgroundTask: BackgroundTasks
):
    input = [data.sepalLength, data.sepalWidth, data.petalLength, data.petalWidth]
    predictions = models["logistic_regression"].predict([input]).tolist()
    output_log = {
        "sepalLength": data.sepalLength,
        "sepalWidth": data.sepalWidth,
        "petalLength": data.petalLength,
        "petalWidth": data.petalWidth,
        "species": classes[predictions[0]],
        "model":"logistic regression"
    }
    backgroundTask.add_task(write_prediction_into_file, output_log)
    # await asyncio.sleep(delay)
    return {"predictions": classes[predictions[0]]}


@app.post("/predict/random_forest")
async def predict_by_random_forest(data: IrisData, backgroundTask: BackgroundTasks):
    input = [data.sepalLength, data.sepalWidth, data.petalLength, data.petalWidth]
    predictions = models["random_forest"].predict([input]).tolist()
    output_log = {
        "sepalLength": data.sepalLength,
        "sepalWidth": data.sepalWidth,
        "petalLength": data.petalLength,
        "petalWidth": data.petalWidth,
        "species": classes[predictions[0]],
        "model":"Random Forest"
    }
    backgroundTask.add_task(write_prediction_into_file, output_log)
    # await asyncio.sleep(delay2)
    return {"predictions": classes[predictions[0]]}


@app.get("/models")
async def get_models():
    print(LOGISTIC_REGRESSION)
    print(RANDOM_FOREST)
    return {"models": list(models.keys())}


@app.get("/")
async def root():
    return {"message": "AMMI Program - Mlops Lab"}
