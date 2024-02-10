from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import uuid
from typing import Optional

app = FastAPI()

class CodeInput(BaseModel):
    code: str
    timeout: Optional[int] = 5  # Default timeout to 5 seconds

@app.post("/execute/")
async def execute_code(code_input: CodeInput):
    # Create a unique file for each execution request to minimize conflict
    file_name = f"temp_code_{uuid.uuid4()}.py"
    try:
        # Write the user's code to a temporary file
        with open(file_name, "w") as code_file:
            code_file.write(code_input.code)

        # Attempt to execute the Python code using subprocess
        # Use a timeout to prevent long-running code
        result = subprocess.run(["python", file_name], capture_output=True, text=True, timeout=code_input.timeout)

        # Clean up by removing the temporary file
        subprocess.run(["rm", file_name])

        if result.stderr:
            return {"error": result.stderr}
        else:
            return {"output": result.stdout}
    except subprocess.TimeoutExpired:
        return {"error": "Execution time exceeded the limit."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
