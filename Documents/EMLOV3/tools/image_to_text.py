from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from typing import List
import uvicorn

app = FastAPI()

# Initialize the Hugging Face pipeline for the specific multimodal model
model_pipeline = pipeline("image-to-text", model="vikhyatk/moondream1")

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    # The model expects a list of images, but here we process only one image at a time
    try:
        # Generate a description from the image
        result = model_pipeline(images=[contents])
        return {"description": result[0]['generated_text']}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
