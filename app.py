from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from io import StringIO

app = FastAPI()

# Enable CORS (you can restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and tokenizer
model_path = "./final_model/final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

label_map = {0: "hatespeech", 1: "not_hatespeech"}

@app.post("/Hatespeech_detection-csv")
async def hatespeech_detection_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print("📄 CSV Loaded:", df.head())

        if "text" not in df.columns:
            return JSONResponse(content={"error": "CSV must contain a 'text' column."}, status_code=400)

        # Tokenize and predict
        inputs = tokenizer(df["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        df["Detection_result"] = [label_map[p] for p in preds]

        print("✅ Predictions:", df["Detection_result"].value_counts().to_dict())

        # Convert to downloadable CSV
        output = StringIO()
        df[["text", "Detection_result"]].to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/Hatespeech_detection-json")
async def hatespeech_detection_json(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print("📄 CSV Loaded:", df.head())

        if "text" not in df.columns:
            return JSONResponse(content={"error": "CSV must contain a 'text' column."}, status_code=400)

        # Tokenize and predict
        inputs = tokenizer(df["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        df["Detection_result"] = [label_map[p] for p in preds]

        print("✅ Predictions:", df["Detection_result"].value_counts().to_dict())

        # Return JSON response
        result_json = df[["text", "Detection_result"]].to_dict(orient="records")
        return JSONResponse(content={"predictions": result_json})

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/Hatespeech_detection-text")
async def hatespeech_detection_text(text: str = Form(...)):
    try:
        print("🔤 Text Received:", text)

        # Tokenize and predict
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        result = label_map[pred]

        print("✅ Single prediction:", result)
        return {"text": text, "Detection_result": result}

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
