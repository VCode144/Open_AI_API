from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import openai
import spacy
import io
import sys
import contextlib
import os
from dotenv import load_dotenv

load_dotenv() 
openai.api_key = os.getenv('OPENAI_API_KEY')

nlp = spacy.load("en_core_web_sm")

app = FastAPI()

class UserInput(BaseModel):
    message: str
    user_input: Optional[str] = None

def extract_keywords(text: str):
    doc = nlp(text)
    return [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]

def detect_intent(text: str):
    lowered = text.lower()
    if any(x in lowered for x in ["explain", "describe", "what does this code do"]):
        return "code_explanation"
    elif "generate" in lowered or "write code" in lowered:
        return "code_generation"
    elif "debug" in lowered or "fix" in lowered:
        return "code_debug"
    elif "run" in lowered or "execute" in lowered or "output" in lowered:
        return "code_execution"
    else:
        return "general_query"

def get_gpt_response(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're a helpful AI assistant skilled in code, debugging, and explaining code."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

@contextlib.contextmanager
def capture_stdout():
    new_out = io.StringIO()
    old_out = sys.stdout
    sys.stdout = new_out
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_out

def run_user_code(code: str, user_input: str = ""):
    input_func = lambda _: user_input
    global_vars = {"input": input_func}
    with capture_stdout() as out:
        try:
            exec(code, global_vars)
            return {"success": True, "output": out.getvalue()}
        except Exception as e:
            return {"success": False, "error": str(e)}

@app.post("/process")
async def process_input(user_input: UserInput):
    message = user_input.message
    user_code_input = user_input.user_input or ""
    intent = detect_intent(message)
    keywords = extract_keywords(message)

    if intent == "code_generation":
        prompt = f"Generate Python code for the following task:\n{message}"
        response = get_gpt_response(prompt)
        return {
            "intent": intent,
            "keywords": keywords,
            "response_type": "code",
            "response": response
        }

    elif intent == "code_debug":
        prompt = f"Debug this Python code and explain what's wrong with it:\n{message}"
        response = get_gpt_response(prompt)
        return {
            "intent": intent,
            "keywords": keywords,
            "response_type": "debug",
            "response": response
        }

    elif intent == "code_execution":
        result = run_user_code(message, user_code_input)
        return {
            "intent": intent,
            "keywords": keywords,
            "response_type": "execution",
            **result
        }

    elif intent == "code_explanation":
        prompt = f"Explain what the following Python code does, step-by-step:\n{message}"
        response = get_gpt_response(prompt)
        return {
            "intent": intent,
            "keywords": keywords,
            "response_type": "explanation",
            "response": response
        }

    else:
        response = get_gpt_response(message)
        return {
            "intent": "general_query",
            "keywords": keywords,
            "response_type": "text",
            "response": response
        }
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
