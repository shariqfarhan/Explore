from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

app = FastAPI()

class Content(BaseModel):
    text: str

# Assuming you have set your OpenAI API key as an environment variable,
# you can also set it directly in your code with openai.api_key = "your-api-key"

@app.post("/check-safety/")
async def check_safety(content: Content):
    """Check if the content is safe using GPT."""
    try:
        # Use the OpenAI API to classify the text content
        response = openai.Completion.create(
            engine="text-davinci-003",  # You might use a different engine based on availability and requirements
            prompt=f"Is the following text safe for all audiences? Provide detailed reasons.\n\nText: \"{content.text}\"",
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # Analyze the response to determine if the content is safe
        is_safe = "no" not in response.choices[0].text.lower()
        reason = response.choices[0].text.strip() if not is_safe else "Content is considered safe"

        return {"is_safe": is_safe, "reason": reason}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
