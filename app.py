"""
Neural-Loom — Adaptive AI Learning Platform
============================================
All backend logic lives here (Flask + MongoDB + Gemini + Groq).
Eye-tracking engagement signals arrive via a JSON endpoint from the browser.
"""

import os
import json
import re
import threading
from datetime import datetime
from functools import wraps
import urllib.parse

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId
import google.generativeai as genai
from groq import Groq

def fully_unquote(text):
    if not text: return ""
    prev = ""
    while text != prev:
        prev = text
        text = urllib.parse.unquote(text)
    return text

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/neuralloom")
client = MongoClient(MONGO_URI)
db = client.neuralloom

users_col       = db.users
courses_col     = db.courses
lessons_col     = db.lessons
quizzes_col     = db.quizzes
engagement_col  = db.engagement_logs

# ---------------------------------------------------------------------------
# AI Clients
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def current_user():
    if "user_id" not in session:
        return None
    return users_col.find_one({"_id": ObjectId(session["user_id"])})

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
def gemini_ask(prompt: str) -> str:
    """Send a prompt to Groq (formerly Gemini) and return the text response."""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI error: {e}]"


def analyze_syllabus(subject: str, syllabus_text: str) -> dict:
    """
    Ask Gemini to parse the syllabus into structured modules.
    Returns: {"modules": [{"title": str, "topics": [str]}]}
    """
    prompt = f"""
You are a curriculum designer. Analyze this syllabus for "{subject}" and
structure it into learning modules. Return ONLY valid JSON, no markdown fences.

Format:
{{
  "modules": [
    {{"title": "Module title", "topics": ["Topic A", "Topic B"]}}
  ]
}}

Syllabus:
{syllabus_text}
"""
    raw = gemini_ask(prompt)
    # Strip possible markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        # Fallback: single module with raw text
        return {"modules": [{"title": subject, "topics": [syllabus_text[:200]]}]}


def generate_lesson(subject: str, topic: str, style: str = "normal") -> str:
    """
    Generate lesson content via Gemini.
    style: "normal" | "simplified" | "story" | "interactive"
    """
    style_instructions = {
        "normal": "Explain clearly with examples and step-by-step breakdown.",
        "simplified": (
            "The student is confused. Simplify this topic drastically. "
            "Use short sentences, a real-world analogy, and a worked example. Avoid jargon. "
            "Most importantly, include highly visual HTML elements where possible: "
            "1. Concept comparisons: `<div class='flip-card'><div class='flip-card-inner'><div class='flip-card-front'>Concept</div><div class='flip-card-back'>Simple explanation</div></div></div>` "
            "2. Step-by-step processes: `<div class='timeline'><div class='timeline-item'>Step 1...</div></div>` "
            "Do NOT use double newlines inside these HTML blocks."
        ),
        "story": (
            "The student is bored. Make this lesson a short, gripping story or "
            "narrative scenario where the student is the hero solving a real problem "
            "using this concept. Be vivid and engaging. Include the concept naturally."
        ),
        "interactive": (
            "The student is disengaged. Transform this lesson into a challenge or mission. "
            "Present it as a problem to solve, a mystery to crack, or a mission to complete. "
            "Make it exciting. Include a clear objective, the concept needed to solve it, "
            "and a step-by-step challenge."
        ),
    }
    instruction = style_instructions.get(style, style_instructions["normal"])
    prompt = f"""
You are a brilliant, engaging tutor teaching "{subject}".
Topic: {topic}

{instruction}

Format your response with clear headings using **bold** text.
Keep it under 600 words unless a thorough explanation is needed.
"""
    return gemini_ask(prompt)


def generate_analogy(subject: str, topic: str) -> str:
    prompt = f"""
Give one creative, memorable analogy that explains "{topic}" in "{subject}"
to a student who is struggling. Keep it under 100 words. Start directly with the analogy.
"""
    return gemini_ask(prompt)

# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------
def generate_quiz(subject: str, topic: str, num_questions: int = 5) -> list:
    """
    Use Groq to generate MCQ quiz questions.
    Returns list of dicts: {question, options[4], answer, explanation}
    """
    prompt = f"""
Generate {num_questions} multiple-choice quiz questions for the topic "{topic}"
in the subject "{subject}".

Return ONLY a JSON array, no markdown fences. Format:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "..."
  }}
]
"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        return [{
            "question": f"Error generating quiz: {e}",
            "options": ["A) N/A", "B) N/A", "C) N/A", "D) N/A"],
            "answer": "A",
            "explanation": "Please try again."
        }]


def generate_challenge(subject: str, topic: str) -> dict:
    """
    Use Groq to generate an interactive challenge for bored students.
    Returns: {mission, scenario, task, hint, solution}
    """
    prompt = f"""
Create an exciting challenge/mission for a bored student studying "{topic}" in "{subject}".
Make it feel like a game or adventure. Return ONLY JSON, no markdown.

Format:
{{
  "mission": "Short mission title (≤10 words)",
  "scenario": "2-3 sentence dramatic scenario that sets the scene",
  "task": "The specific problem they must solve using the concept",
  "hint": "A helpful hint",
  "solution": "The correct approach / answer"
}}
"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=512,
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception:
        return {
            "mission": "Mission: Master the Concept",
            "scenario": f"You must use your knowledge of {topic} to solve a critical problem.",
            "task": f"Apply {topic} to solve the challenge.",
            "hint": "Think about the core definition.",
            "solution": "Review the lesson content for the answer."
        }

# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if users_col.find_one({"email": email}):
            flash("Email already registered.", "danger")
            return render_template("register.html")

        users_col.insert_one({
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.utcnow(),
        })
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user     = users_col.find_one({"email": email})

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"]   = str(user["_id"])
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------------------------------------------------------------
# Routes — Dashboard & Courses
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    uid     = session["user_id"]
    courses = list(courses_col.find({"user_id": uid}).sort("created_at", -1))
    return render_template("dashboard.html", courses=courses, user_name=session["user_name"])


@app.route("/course/new", methods=["GET", "POST"])
@login_required
def new_course():
    if request.method == "POST":
        subject  = request.form.get("subject", "").strip()
        syllabus = request.form.get("syllabus", "").strip()
        books    = request.form.get("books", "").strip()

        if not subject or not syllabus:
            flash("Subject and syllabus are required.", "danger")
            return render_template("new_course.html")

        # Parse syllabus with Gemini
        structure = analyze_syllabus(subject, syllabus)

        course_id = courses_col.insert_one({
            "user_id":    session["user_id"],
            "subject":    subject,
            "syllabus":   syllabus,
            "books":      books,
            "structure":  structure,
            "created_at": datetime.utcnow(),
        }).inserted_id

        flash("Course created! Lessons are ready.", "success")
        return redirect(url_for("course", course_id=str(course_id)))

    return render_template("new_course.html")


@app.route("/course/<course_id>")
@login_required
def course(course_id):
    c = courses_col.find_one({"_id": ObjectId(course_id), "user_id": session["user_id"]})
    if not c:
        flash("Course not found.", "danger")
        return redirect(url_for("dashboard"))

    # Build flat topic list with module context
    topics = []
    for mod in c.get("structure", {}).get("modules", []):
        for topic in mod.get("topics", []):
            topics.append({"module": mod["title"], "topic": topic})

    return render_template("course.html", course=c, topics=topics)


@app.route("/course/<course_id>/delete", methods=["POST"])
@login_required
def delete_course(course_id):
    courses_col.delete_one({"_id": ObjectId(course_id), "user_id": session["user_id"]})
    lessons_col.delete_many({"course_id": course_id})
    quizzes_col.delete_many({"course_id": course_id})
    flash("Course deleted.", "info")
    return redirect(url_for("dashboard"))

# ---------------------------------------------------------------------------
# Routes — Lessons
# ---------------------------------------------------------------------------
@app.route("/course/<course_id>/lesson")
@login_required
def lesson(course_id):
    topic   = fully_unquote(request.args.get("topic", ""))
    style   = request.args.get("style", "normal")
    c       = courses_col.find_one({"_id": ObjectId(course_id)})

    if not c or not topic:
        flash("Invalid lesson request.", "danger")
        return redirect(url_for("dashboard"))

    # Check cache
    cached = lessons_col.find_one({"course_id": course_id, "topic": topic, "style": style})
    if cached:
        content = cached["content"]
    else:
        content = generate_lesson(c["subject"], topic, style)
        lessons_col.insert_one({
            "course_id":  course_id,
            "topic":      topic,
            "style":      style,
            "content":    content,
            "created_at": datetime.utcnow(),
        })

    analogy = None
    if style == "simplified":
        analogy = generate_analogy(c["subject"], topic)

    # Convert **bold** markdown to <strong> for display
    content_html = markdown_to_html(content)

    return render_template(
        "lesson.html",
        course=c,
        topic=topic,
        style=style,
        content_html=content_html,
        analogy=analogy,
        course_id=course_id,
    )


def markdown_to_html(text: str) -> str:
    """Minimal markdown-to-HTML: bold, line breaks, headers, lists, and HTML block pass-through."""
    # Protect HTML blocks by parsing them out first, or just ensuring they don't break.
    # We will just rely on the AI not using double newlines in HTML blocks.

    # Headers
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Bullet lists
    text = re.sub(r"^\s*[-•]\s+(.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"(<li>.*?</li>(\n|$))+", lambda m: "<ul>" + m.group(0) + "</ul>", text, flags=re.DOTALL)
    # Paragraphs (double newline)
    parts = re.split(r"\n{2,}", text)
    result = []
    for part in parts:
        part = part.strip()
        if part and not part.startswith("<"):
            part = "<p>" + part.replace("\n", "<br>") + "</p>"
        result.append(part)
    return "\n".join(result)

# ---------------------------------------------------------------------------
# Routes — Quiz
# ---------------------------------------------------------------------------
@app.route("/course/<course_id>/quiz")
@login_required
def quiz(course_id):
    topic = fully_unquote(request.args.get("topic", ""))
    c     = courses_col.find_one({"_id": ObjectId(course_id)})
    if not c or not topic:
        flash("Invalid quiz request.", "danger")
        return redirect(url_for("dashboard"))

    cached = quizzes_col.find_one({"course_id": course_id, "topic": topic})
    if cached:
        questions = cached["questions"]
    else:
        questions = generate_quiz(c["subject"], topic)
        quizzes_col.insert_one({
            "course_id":  course_id,
            "topic":      topic,
            "questions":  questions,
            "created_at": datetime.utcnow(),
        })

    return render_template("quiz.html", course=c, topic=topic, questions=questions, course_id=course_id)

# ---------------------------------------------------------------------------
# Routes — Interactive / Challenge Mode
# ---------------------------------------------------------------------------
@app.route("/course/<course_id>/challenge")
@login_required
def challenge(course_id):
    topic = fully_unquote(request.args.get("topic", ""))
    c     = courses_col.find_one({"_id": ObjectId(course_id)})
    if not c or not topic:
        flash("Invalid challenge request.", "danger")
        return redirect(url_for("dashboard"))

    ch_data = generate_challenge(c["subject"], topic)
    # Also generate a story-style lesson for the challenge page
    story_content = generate_lesson(c["subject"], topic, style="interactive")
    story_html    = markdown_to_html(story_content)

    return render_template(
        "interactive.html",
        course=c,
        topic=topic,
        challenge=ch_data,
        story_html=story_html,
        course_id=course_id,
    )

# ---------------------------------------------------------------------------
# Routes — Engagement API (called by camera.js)
# ---------------------------------------------------------------------------
@app.route("/api/engagement", methods=["POST"])
@login_required
def log_engagement():
    """
    Receives engagement state from the browser-side eye tracker.
    Payload: { course_id, topic, state: "focused"|"confused"|"bored" }
    Never stores raw video — only the classified state + timestamp.
    """
    data      = request.get_json(force=True)
    course_id = data.get("course_id", "")
    topic     = data.get("topic", "")
    state     = data.get("state", "focused")

    engagement_col.insert_one({
        "user_id":   session["user_id"],
        "course_id": course_id,
        "topic":     topic,
        "state":     state,
        "timestamp": datetime.utcnow(),
    })

    # Decide what action to take based on state
    action = "continue"
    redirect_url = None

    if state == "confused":
        action = "simplify"
        redirect_url = url_for("lesson", course_id=course_id, topic=topic, style="simplified")
    elif state == "bored":
        action = "challenge"
        redirect_url = url_for("challenge", course_id=course_id, topic=topic)

    return jsonify({"action": action, "redirect": redirect_url})


@app.route("/api/engagement/stats/<course_id>")
@login_required
def engagement_stats(course_id):
    """Return engagement distribution for a course (for analytics)."""
    logs  = list(engagement_col.find({"course_id": course_id, "user_id": session["user_id"]}))
    stats = {"focused": 0, "confused": 0, "bored": 0}
    for log in logs:
        s = log.get("state", "focused")
        stats[s] = stats.get(s, 0) + 1
    return jsonify(stats)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
