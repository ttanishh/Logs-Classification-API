import os
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse

from classify_logic import classify

app = FastAPI()

# Ensure the output directory exists
os.makedirs("resources", exist_ok=True)

@app.post("/classify/")
async def classify_logs(file: UploadFile):
    """
    Endpoint to classify uploaded CSV logs.

    Expects a CSV file with columns: 'source' and 'log_message'
    Returns a downloadable CSV with an added 'target_label' column.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")

    try:
        # Read uploaded CSV
        df = pd.read_csv(file.file)

        # Validate required columns
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        # Perform classification using logic module
        df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

        # Save the result
        output_path = "resources/output.csv"
        df.to_csv(output_path, index=False)

        return FileResponse(output_path, media_type='text/csv', filename="classified_logs.csv")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    finally:
        file.file.close()
