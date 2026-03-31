# Neural-Loom — Adaptive AI Learning Platform

> A personal AI tutor that watches how you learn and changes how it teaches — in real time.

---

## What It Does

Neural-Loom is a Flask-based adaptive learning platform where:

1. **You upload a syllabus** → Gemini AI structures it into modules and topics
2. **Lessons are generated on demand** → Gemini writes explanations, examples, analogies
3. **Quizzes are auto-created** → Groq LLM generates MCQ questions per topic
4. **Your webcam tracks your eyes** → MediaPipe + OpenCV classify your engagement
5. **The UI adapts automatically**:
   - Confused? → Simplified explanation + analogy
   - Bored? → Mission/Challenge mode with gamified learning

---

## Project Structure

```
neural-loom/
├── app.py                          ← All Flask backend logic
├── requirements.txt
├── .env.example
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── new_course.html
│   ├── course.html
│   ├── lesson.html
│   ├── quiz.html
│   └── interactive.html
├── static/
│   ├── css/style.css
│   └── js/camera.js
└── models/
    └── eye_tracking_model/
        └── tracker.py
```

---

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure .env

```bash
cp .env.example .env
# Fill in: GEMINI_API_KEY, GROQ_API_KEY, MONGO_URI, SECRET_KEY
```

- Gemini key: https://aistudio.google.com/app/apikey
- Groq key:   https://console.groq.com/keys

### 3. Start MongoDB

```bash
# Docker (easiest)
docker run -d -p 27017:27017 mongo
```

### 4. Run

```bash
python app.py
# Open http://localhost:5000
```

---

## How Engagement Detection Works

The browser loads MediaPipe FaceMesh via CDN. Every frame from the webcam is processed **locally** — no video ever leaves the device.

- **Eye Aspect Ratio (EAR)** measures how open the eyes are → detects blinks
- **Iris position ratio** measures gaze direction → detects looking away
- Over a 10-second sliding window:
  - High blink rate or frequent gaze aversion → **Bored** → Challenge mode
  - Very low blink rate + low EAR → **Confused** → Simplified lesson
  - Otherwise → **Focused** → Continue
- Every 15 seconds only the state string is POSTed to `/api/engagement`

---

## Teaching Modes

| Mode | Trigger | What happens |
|---|---|---|
| Standard | Default | Full explanation with examples |
| Simplified | Eye tracking: confused | Short sentences, analogy, bullet breakdown |
| Story | Manual | Narrative scenario lesson |
| Interactive | Eye tracking: bored | Mission card, gamified challenge, XP counter |

---

## Privacy

Raw webcam frames are never stored or transmitted.
Only the classified state string is sent to the server.
All biometric processing runs locally via MediaPipe WASM.

---

## API Routes

| Route | Purpose |
|---|---|
| `POST /api/engagement` | Receive state from browser tracker |
| `GET /api/engagement/stats/<id>` | Engagement breakdown per course |
| `GET /course/<id>/lesson?topic=X&style=Y` | Lesson (normal/simplified/story/interactive) |
| `GET /course/<id>/quiz?topic=X` | MCQ quiz |
| `GET /course/<id>/challenge?topic=X` | Mission/challenge mode |

---

## Notes on Python Version

mediapipe supports up to Python 3.12. On Python 3.13/3.14:
- Remove `mediapipe` and `opencv-python` from requirements.txt
- The server-side `tracker.py` will be unavailable
- `camera.js` handles all tracking in the browser and works on any Python version

