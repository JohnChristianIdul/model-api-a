from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging
import io
import traceback

from app.model.model import load_model, predict_pipeline

app = FastAPI(
    title="Water Level Prediction API fo Basin A",
    description="API for predicting water levels using TCN model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mtcn-informer.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Startup failed to load model: {e}")


@app.get("/")
def root():
    return {"message": "Water Level Prediction API is running"}


@app.post("/predict/wl-a")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")

        content = await file.read()

        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            logger.error(f"File parsing error: {e}")
            raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

        if 'Datetime' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing required column: Datetime")

        result = predict_pipeline(df)

        if "error" in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Unhandled error in prediction: {str(e)}\nTraceback: {tb}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
