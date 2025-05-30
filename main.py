import csv
import base64
import time
import json
import traceback
from io import BytesIO
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from openai import OpenAI
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Initialize FastAPI app
app = FastAPI()

# Global lectures store
lectures = []

# --- Middleware for catching all errors ---
class CatchAllMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except RequestValidationError as ve:
            print("[VALIDATION ERROR]", ve.errors())
            return JSONResponse(
                status_code=422,
                content={"detail": ve.errors()}
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] Unhandled Exception: {e}\n{tb}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error", "details": str(e)}
            )

app.add_middleware(CatchAllMiddleware)

# --- Pydantic model for request ---
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

# --- Load lecture content ---
def load_lectures_csv(file_path="tds_lectures_content.csv"):
    global lectures
    lectures.clear()
    print(f"[DEBUG] Loading lectures from {file_path}")
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lectures.append({
                "title": row["Lecture Title"],
                "content": row["Content"]
            })
    print(f"[DEBUG] Loaded {len(lectures)} lectures")

@app.on_event("startup")
async def startup_event():
    load_lectures_csv()

# --- OCR ---
def extract_text_from_image(base64_str: str) -> str:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        print(f"[DEBUG] OCR extracted text: {text.strip()}")
        return text
    except Exception as e:
        print("[ERROR] OCR failed:", e)
        return ""

# --- Lecture search ---
def find_relevant_lectures(question: str, limit=3) -> List[dict]:
    question_lower = question.lower()
    matches = [lec for lec in lectures if question_lower in lec["content"].lower()]
    print(f"[DEBUG] Found {len(matches)} matching lectures")
    return matches[:limit] if matches else lectures[:limit]

# --- GPT call ---
def call_gpt(question: str, context: str, retries=4) -> str:
    prompt = f"{context}\n\nQ: {question}\nA:"
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] GPT retry {attempt + 1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    return "ERROR: Failed to get response from GPT after retries."

# --- Core question processing ---
def process_question(question, image):
    if not question:
        raise ValueError("Missing 'question' in request.")

    # Step 1: OCR from base64 image
    ocr_text = ""
    if image:
        try:
            ocr_text = extract_text_from_image(image)
            question += "\n\n" + ocr_text
        except Exception as e:
            print("[WARNING] Image OCR failed:", e)

    # Step 2: Find relevant lectures
    relevant_lectures = find_relevant_lectures(question)
    context = "\n\n".join([f"{lec['title']}:\n{lec['content']}" for lec in relevant_lectures])

    # Step 3: Call GPT
    answer_text = call_gpt(question, context)

    # Step 4: Dummy links
    links = []
    for lec in relevant_lectures:
        if "ga4" in lec["content"].lower():
            links.append({
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959",
                "text": "GA4 Discussion"
            })
        elif "gpt-3.5" in question.lower():
            links.append({
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
                "text": "GA5 Clarification"
            })

    return {
        "answer": answer_text,
        "links": links
    }

# --- Main API endpoint ---
@app.post("/api")
async def answer_question(data: QuestionRequest):
    try:
        answer = process_question(data.question, data.image)
        return JSONResponse(content=answer)
    except Exception as e:
        print("[ERROR] API processing failed:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

# --- Health check ---
@app.get("/")
async def health_check():
    return {"message": "TDS Virtual TA API is running. Use POST /api to ask questions."}
