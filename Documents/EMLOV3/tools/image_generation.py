from fastapi import FastAPI, HTTPException
from transformers import pipeline
from diffusers import AutoPipelineForText2Image
import torch
import uvicorn

app = FastAPI()

# Initialize the pipeline with the multimodal LLM model
if torch.cuda.is_available():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
else:
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp32")
    pipe.to("cpu")

@app.post("/generate-image/")
async def generate_image(prompt: str):
    try:
        # Generate text based on the user's prompt
        image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        return {"image": image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
