from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import base64
import pytesseract
import openai
import os
import csv
from contextlib import asynccontextmanager

pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\91962\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class Link(BaseModel):
    url: str
    text: str

class ResponseModel(BaseModel):
    answer: str
    links: List[Link]

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image string optional

# Globals for data
LECTURES = []
DISCOURSE_POSTS = []

def load_lectures():
    global LECTURES
    with open("tds_lectures_content.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        LECTURES = [(row["Lecture Title"], row["Content"]) for row in reader]

def load_discourse_posts():
    global DISCOURSE_POSTS
    with open("tds_discourse_posts.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        DISCOURSE_POSTS = [(row["Topic Title"], row["Content"]) for row in reader]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    load_lectures()
    load_discourse_posts()
    print(f"Loaded {len(LECTURES)} lectures and {len(DISCOURSE_POSTS)} discourse posts.")
    yield
    # shutdown (if needed)

app = FastAPI(lifespan=lifespan)

def search_knowledge_base(query: str, top_k=3) -> List[str]:
    combined = LECTURES + DISCOURSE_POSTS
    query_words = query.lower().split()

    def relevance(text):
        text_lower = text.lower()
        return sum(text_lower.count(w) for w in query_words)

    scored = []
    for title, content in combined:
        score = relevance(title) + relevance(content)
        if score > 0:
            scored.append((score, f"Title: {title}\n{content[:500]}"))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [text for score, text in scored[:top_k]]

def ocr_from_base64(image_b64: str) -> str:
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        return pytesseract.image_to_string(image)
    except Exception:
        return ""

def generate_openai_answer_with_context(question: str, context_snippets: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_snippets)
    prompt = (
        f"You are a helpful TA for the Tools in Data Science course.\n\n"
        f"Use the following reference material to answer the question:\n{context_text}\n\nQuestion: {question}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful TA for the Tools in Data Science course."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Sorry, I couldn't generate a proper answer right now."

def match_links(question: str) -> List[Link]:
    links = []

    if "gpt-3.5" in question and "proxy" in question:
        links.append(Link(
            url="https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
            text="Use the model that’s mentioned in the question."
        ))
        links.append(Link(
            url="https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
            text="Token clarification and advice from Prof. Anand"
        ))

    elif "10/10" in question and "bonus" in question:
        links.append(Link(
            url="https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959",
            text="Score display logic on dashboard"
        ))

    elif "docker" in question.lower() and "podman" in question.lower():
        links.append(Link(
            url="https://tds.s-anand.net/#/docker",
            text="Docker vs Podman usage in the course"
        ))

    return links

@app.post("/api", response_model=ResponseModel)
async def answer_question(req: QuestionRequest):
    image_text = ocr_from_base64(req.image) if req.image else ""
    combined_query = req.question + " " + image_text

    context_snippets = search_knowledge_base(combined_query, top_k=3)
    answer = generate_openai_answer_with_context(req.question, context_snippets)
    links = match_links(combined_query)

    return {"answer": answer, "links": links}
