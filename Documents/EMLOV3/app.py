from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from main_llm.main_llm import handle_query
from tools import web_search, image_generation, code_interpreter, image_to_text
from memory.memory_manager import MemoryManager
from safety.content_safety import is_content_safe

app = FastAPI()
memory_manager = MemoryManager()

class QueryInput(BaseModel):
    query: str
    context: Optional[str] = ""

@app.post("/process-query/")
async def process_query_endpoint(query_input: QueryInput):
    context = memory_manager.get_memory()

    # If context is provided in the request, use it; otherwise, use the memory context
    final_context = query_input.context if query_input.context else context

    llm_response = handle_query(query_input.query, final_context)

    # Logic to determine the action based on LLM response or query itself
    if "search:" in llm_response:
        search_query = llm_response.split("search:",1)[1].strip()
        response = web_search(search_query)
    elif "generate image:" in llm_response:
        image_prompt = llm_response.split("generate image:",1)[1].strip()
        response = image_generation(image_prompt)
    elif "interpret code:" in llm_response:
        code_snippet = llm_response.split("interpret code:",1)[1].strip()
        response = code_interpreter(code_snippet)
    elif "image to text:" in llm_response:
        image_url = llm_response.split("image to text:",1)[1].strip()
        response = image_to_text(image_url)
    else:
        response = llm_response

    # Check if the generated content is safe
    if is_content_safe(response):
        memory_manager.add_to_memory(f"Query: {query_input.query} | Response: {response}")
        return {"response": response}
    else:
        return {"response": "I'm sorry, I can't provide a response to that. Please try a different query."}
