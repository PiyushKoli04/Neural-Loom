/**
 * Neural-Loom — camera.js
 * ========================
 * Browser-side eye tracking using MediaPipe FaceMesh (via CDN).
 * Runs entirely client-side. RAW FRAMES ARE NEVER SENT TO THE SERVER.
 * Only the classified engagement state string is posted to /api/engagement.
 *
 * How it works:
 *  1. Request webcam access (user must consent).
 *  2. Run MediaPipe FaceMesh on each frame locally.
 *  3. Compute Eye Aspect Ratio (EAR) + gaze direction locally.
 *  4. Classify engagement: "focused" | "confused" | "bored"
 *  5. POST only the state string + course/topic metadata every 15 seconds.
 *  6. If server returns a redirect action, navigate there.
 */

(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Config (injected by Flask template via data attributes on <body>)
  // -------------------------------------------------------------------------
  const body       = document.body;
  const COURSE_ID  = body.dataset.courseId  || "";
  const TOPIC      = body.dataset.topic     || "";
  const REPORT_MS  = 15_000;   // report to server every 15 s
  const WINDOW_S   = 10;       // sliding window for metrics (seconds)

  // EAR thresholds (mirroring Python tracker)
  const EAR_BLINK   = 0.25;
  const EAR_DROWSY  = 0.28;
  const BLINK_FRAMES = 2;
  const GAZE_OFF_THRESHOLD = 0.30;
  const BORED_BLINK_RATE   = 25;  // blinks/min
  const CONFUSED_BLINK_RATE = 5;

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  let videoEl   = null;
  let canvasEl  = null;
  let ctx       = null;
  let faceMesh  = null;
  let camera    = null;

  let earHistory       = [];   // {t, ear}
  let blinkTimes       = [];   // timestamps
  let gazeOffHistory   = [];   // {t, off}
  let blinkCounter     = 0;
  let currentState     = "focused";
  let reportTimer      = null;

  // -------------------------------------------------------------------------
  // Math helpers
  // -------------------------------------------------------------------------
  function dist2D(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function ear(lm, topI, botI, leftI, rightI) {
    const vert  = dist2D(lm[topI], lm[botI]);
    const horiz = dist2D(lm[leftI], lm[rightI]);
    return horiz === 0 ? 0 : vert / horiz;
  }

  // Iris X ratio [0..1] within the eye horizontal span
  function irisRatio(lm, irisIndices, leftI, rightI) {
    const xs     = irisIndices.map(i => lm[i].x);
    const irisX  = xs.reduce((a, b) => a + b, 0) / xs.length;
    const eyeL   = lm[leftI].x;
    const eyeR   = lm[rightI].x;
    const w      = eyeR - eyeL;
    return w === 0 ? 0.5 : (irisX - eyeL) / w;
  }

  // MediaPipe Face Mesh landmark indices
  const IDX = {
    L_TOP: 159, L_BOT: 145, L_LEFT: 33,  L_RIGHT: 133,
    R_TOP: 386, R_BOT: 374, R_LEFT: 362, R_RIGHT: 263,
    L_IRIS: [468, 469, 470, 471, 472],
    R_IRIS: [473, 474, 475, 476, 477],
  };

  // -------------------------------------------------------------------------
  // Classifier
  // -------------------------------------------------------------------------
  function classify() {
    const now    = Date.now() / 1000;
    const cutoff = now - WINDOW_S;

    // Trim old entries
    earHistory      = earHistory.filter(e => e.t >= cutoff);
    blinkTimes      = blinkTimes.filter(t => t >= cutoff);
    gazeOffHistory  = gazeOffHistory.filter(e => e.t >= cutoff);

    if (earHistory.length === 0) return "focused";

    const blinkRate  = blinkTimes.length * (60 / WINDOW_S);
    const meanEar    = earHistory.reduce((s, e) => s + e.ear, 0) / earHistory.length;
    const gazeOffFrc = gazeOffHistory.length > 0
      ? gazeOffHistory.filter(e => e.off).length / gazeOffHistory.length
      : 0;

    // If drowsy (eyes droopy), extremely high blink rate, or looking away completely
    if (meanEar < EAR_DROWSY || blinkRate > BORED_BLINK_RATE || gazeOffFrc > 0.6) return "bored";
    
    // If looking back and forth off-screen moderately
    if (gazeOffFrc > 0.3) return "confused";
    
    return "focused";
  }

  // -------------------------------------------------------------------------
  // MediaPipe callback
  // -------------------------------------------------------------------------
  function onResults(results) {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;

    const lm  = results.multiFaceLandmarks[0];
    const now = Date.now() / 1000;

    const leftEar  = ear(lm, IDX.L_TOP, IDX.L_BOT, IDX.L_LEFT, IDX.L_RIGHT);
    const rightEar = ear(lm, IDX.R_TOP, IDX.R_BOT, IDX.R_LEFT, IDX.R_RIGHT);
    const avgEar   = (leftEar + rightEar) / 2;

    // Blink
    if (avgEar < EAR_BLINK) {
      blinkCounter++;
    } else {
      if (blinkCounter >= BLINK_FRAMES) blinkTimes.push(now);
      blinkCounter = 0;
    }

    // Gaze
    let gazeOff = false;
    try {
      const lIris = irisRatio(lm, IDX.L_IRIS, IDX.L_LEFT, IDX.L_RIGHT);
      const rIris = irisRatio(lm, IDX.R_IRIS, IDX.R_LEFT, IDX.R_RIGHT);
      gazeOff = Math.abs((lIris + rIris) / 2 - 0.5) > GAZE_OFF_THRESHOLD;
    } catch (_) {}

    earHistory.push({ t: now, ear: avgEar });
    gazeOffHistory.push({ t: now, off: gazeOff });

    currentState = classify();
    updateIndicator(currentState);
  }

  // -------------------------------------------------------------------------
  // UI indicator
  // -------------------------------------------------------------------------
  function updateIndicator(state) {
    const el = document.getElementById("engagement-indicator");
    if (!el) return;
    const labels = { focused: "🟢 Focused", confused: "🟡 Confused", bored: "🔴 Bored" };
    el.textContent = labels[state] || "⬜ Tracking…";
    el.className   = "engagement-badge engagement-" + state;
  }

  // -------------------------------------------------------------------------
  // Report to server
  // -------------------------------------------------------------------------
  async function reportToServer() {
    if (!COURSE_ID || !TOPIC) return;

    try {
      const res = await fetch("/api/engagement", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          course_id: COURSE_ID,
          topic:     TOPIC,
          state:     currentState,
        }),
      });

      const data = await res.json();

      // If server wants us to change mode, navigate after a short delay
      if (data.redirect && data.action !== "continue") {
        showAdaptiveNotice(data.action, data.redirect);
      }
    } catch (err) {
      console.warn("[Neural-Loom] Engagement report failed:", err);
    }
  }

  // -------------------------------------------------------------------------
  // Adaptive notice overlay
  // -------------------------------------------------------------------------
  function showAdaptiveNotice(action, redirectUrl) {
    const existing = document.getElementById("adaptive-notice");
    if (existing) return;   // already showing

    const messages = {
      simplify:  { title: "Let me explain differently…", sub: "Switching to a simpler explanation just for you.", icon: "💡" },
      challenge: { title: "Mission incoming!", sub: "You look ready for a challenge. Let's make this exciting!", icon: "🚀" },
    };
    const msg = messages[action] || { title: "Adapting…", sub: "", icon: "🔄" };

    const div  = document.createElement("div");
    div.id     = "adaptive-notice";
    div.innerHTML = `
      <div class="adaptive-card">
        <div class="adaptive-icon">${msg.icon}</div>
        <h3>${msg.title}</h3>
        <p>${msg.sub}</p>
        <div class="adaptive-bar"><div class="adaptive-fill"></div></div>
      </div>`;
    document.body.appendChild(div);

    setTimeout(() => {
      window.location.href = redirectUrl;
    }, 3000);
  }

  // -------------------------------------------------------------------------
  // Initialise MediaPipe (loaded via CDN in base.html)
  // -------------------------------------------------------------------------
  function initTracker() {
    // Container for video and button
    const container = document.createElement("div");
    container.style.position = "fixed";
    container.style.bottom = "20px";
    container.style.left = "20px";
    container.style.zIndex = "9999";
    
    // Video element
    videoEl  = document.createElement("video");
    videoEl.id = "camera-overlay-video";
    videoEl.style.width = "200px";
    videoEl.style.border = "3px solid #6366f1";
    videoEl.style.borderRadius = "12px";
    videoEl.style.transform = "scaleX(-1)"; // mirror effect
    videoEl.style.display = "block";
    container.appendChild(videoEl);

    // Camera off button
    const offBtn = document.createElement("button");
    offBtn.innerHTML = "Turn Off Camera ✖";
    offBtn.style.position = "absolute";
    offBtn.style.top = "-12px";
    offBtn.style.right = "-12px";
    offBtn.style.background = "#ef4444";
    offBtn.style.color = "white";
    offBtn.style.border = "none";
    offBtn.style.padding = "4px 8px";
    offBtn.style.borderRadius = "12px";
    offBtn.style.fontSize = "12px";
    offBtn.style.cursor = "pointer";
    offBtn.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
    offBtn.onclick = () => {
      if (camera) camera.stop();
      if (reportTimer) clearInterval(reportTimer);
      container.remove();
      const el = document.getElementById("engagement-indicator");
      if (el) {
        el.textContent = "📷 Camera off";
        el.className = "engagement-badge";
      }
    };
    container.appendChild(offBtn);
    
    document.body.appendChild(container);

    faceMesh = new window.FaceMesh({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults(onResults);

    camera = new window.Camera(videoEl, {
      onFrame: async () => {
        await faceMesh.send({ image: videoEl });
      },
      width: 320,
      height: 240,
    });

    camera.start()
      .then(() => {
        console.log("[Neural-Loom] Eye tracking active.");
        reportTimer = setInterval(reportToServer, REPORT_MS);
      })
      .catch((err) => {
        console.warn("[Neural-Loom] Camera unavailable:", err);
        const el = document.getElementById("engagement-indicator");
        if (el) el.textContent = "📷 Camera off";
      });
  }

  // -------------------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    // Only activate on lesson / challenge pages
    if (!COURSE_ID || !TOPIC) return;

    // Wait for MediaPipe scripts (loaded async in base.html)
    const ready = () => window.FaceMesh && window.Camera;
    if (ready()) {
      initTracker();
    } else {
      const interval = setInterval(() => {
        if (ready()) {
          clearInterval(interval);
          initTracker();
        }
      }, 500);
    }
  });

  // Expose for debugging
  window.NeuralLoom = {
    getState: () => currentState,
    getMetrics: () => ({
      blinkRate: blinkTimes.length * (60 / WINDOW_S),
      earHistory: earHistory.slice(-5),
      state: currentState,
    }),
  };
})();
