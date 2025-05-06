
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./final_model/final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

label_map = {0: "non_hatespeech", 1: "hatespeech"}

@app.post("/Hatespeech_detection-csv")
async def Hatespeech_detection_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print("📄 CSV Loaded:", df.head())  # Debug print

        if "text" not in df.columns:
            return JSONResponse(content={"error": "CSV must contain a 'text' column."}, status_code=400)

        inputs = tokenizer(df["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        df["Detection_result"] = [label_map[p] for p in preds]

        print("✅ Predictions:", df["Detection_result"].value_counts().to_dict())  # Debug print
        return df[["text", "Detection_result"]].to_dict(orient="records")

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


