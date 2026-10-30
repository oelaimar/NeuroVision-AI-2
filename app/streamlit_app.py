"""
app/streamlit_app.py: NeuroVision AI test interface.
 
The app NEVER loads the model: it sends the image to the FastAPI service (POST /predict).
 
Run from the project root (the API must be running):
    streamlit run app/streamlit_app.py
"""

import hashlib
import mimetypes
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
ACCEPTED = ["jpeg", "jpg", "bmp", "png"]
TIMEOUT_S = 30
PALETTE = {0: (230, 57, 70), 1: (244, 162, 97), 2: (42, 157, 143), 3: (67, 97, 238)}   # RGB, one per class id

# talking to API
def get_health(api_url: str) -> Tuple[str, str]:
    """Return (state, text): state is 'ok', 'degraded' or 'down'."""
    try:
        r = requests.get(f"{api_url}/heath", timeout=3)
        data = r.json()
        if data.get("model_loaded"):
            return "ok", f"API ready, model {data.get('model_name')} v{data.get('model_version')} ({data.get('model_alias')})"
        return "degraded", f"API running but model not loaded: {data.get('error')}"

    except requests.RequestException:
        return "down", f"API unreachable at {api_url}"

def call_api(api_url: str, filename: str, data: bytes, conf: float) -> Tuple[Optional[dict], Optional[str]]:
    """POST /predict. Returns (result, None) on success or (None, error message)."""
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    try:
        r = requests.post(f"{api_url}/predict", params={"conf": conf}, files={"file": (filename, data, mime)}, timeout=TIMEOUT_S)
    except requests.ConnectionError:
        return None, (f"The API is unavailable at {api_url}. "
                      "Start it with: uvicorn api.main:app --host 0.0.0.0 --port 8000")
    except requests.Timeout:
        return None, f"The API did not answer within {TIMEOUT_S} seconds."
    except requests.RequestException as e:
        return None, f"Request failed: {e}"

    if r.status_code == 200:
        return r.json(), None
 
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = r.text[:200]
    if r.status_code == 400:
        return None, f"The image was rejected: {detail}"
    if r.status_code == 422:
        return None, f"Invalid request (file or confidence value): {detail}"
    if r.status_code == 503:
        return None, f"The model is not available on the server: {detail}"
    return None, f"Unexpected API error ({r.status_code}): {detail}"


 # Drawing 
def draw_detections(image: np.ndarray, detections: List[dict], api_size: Tuple[int, int]) -> np.ndarray:
    """Draw boxes and 'class score%' labels on a BGR image (OpenCV).
 
    Coordinates come from the API, in the pixels of the image it decoded.
    """
    out = image.copy()
    height, width = out.shape[:2]
    sx, sy = width / api_size[0], height / api_size[1]
    thickness = max(2, width // 200)
    font_scale = max(0.5, width / 1000)

    for d in detections:
        color = PALETTE.get(d["class_id"], (255, 255, 255))[::-1]     # RGB -> BGR for OpenCV
        x1, y1 = int(d["x1"] * sx), int(d["y1"] * sy)
        x2, y2 = int(d["x2"] * sx), int(d["y2"] * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
 
        label = f'{d["class_name"]} {d["confidence"]:.0%}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(y1 - th - baseline - 4, 0)                          # keep the label inside the image
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + baseline + 4), color, -1)
        cv2.putText(out, label, (x1 + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# page

def main() -> None:
    st.set_page_config(page_title="NeuroVision AI", layout="wide")
    st.title("NeuroVision AI: brain tumor detection")
    st.caption("Research prototype, not a medical device. Every result must be checked by a doctor.")

    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("API URL", API_URL).rstrip("/")
        conf = st.sidebar("Confidence threshold", 0.0, 1.0, 0.25, 0.05,
                         help="Lower: more detections (fewer missed tumors, more false alarms). "
                              "Higher: fewer, more certain detections.")
        state, text = get_health(api_url)
        {"ok": st.success, "degraded": st.warning, "down": st.error}[state](text)

        uploaded = st.file_uploader("Upload an MRI image", type=ACCEPTED)
    if uploaded is None:
        st.info("Upload an MRI image (jpeg, jpg, bmp or png) to start.")
        return
 
    data = uploaded.getvalue()
    key = (hashlib.md5(data).hexdigest(), conf, api_url)
    cache = st.session_state.setdefault("results", {})          # successes only: errors are never cached
    if key not in cache:
        with st.spinner("Analysing the image..."):
            result, error = call_api(api_url, uploaded.name, data, conf)
        if error:
            st.error(error)
            return
        cache[key] = result
    result = cache[key]
 
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # BGR, EXIF rotation applied
    if image is None:
        st.error("The image could not be displayed: the file is unreadable.")
        return
 
    detections = result["detections"]
    annotated = draw_detections(image, detections, (result["image_width"], result["image_height"]))
 
    left, right = st.columns(2)
    left.subheader("Original")
    left.image(image, channels="BGR", use_container_width=True)
    right.subheader("Detections")
    right.image(annotated, channels="BGR", use_container_width=True)
 
    if not detections:
        st.info(f"**No tumor detected** at a confidence threshold of {conf:.2f}. "
                "This does not rule out a tumor: try a lower threshold, and ask a doctor to review the scan.")
        return
 
    best = max(d["confidence"] for d in detections)
    c1, c2 = st.columns(2)
    c1.metric("Detections", len(detections))
    c2.metric("Highest confidence", f"{best:.0%}")
    table = pd.DataFrame(detections).rename(columns={"class_name": "class"})
    table["confidence"] = table["confidence"].map("{:.1%}".format)
    st.dataframe(table[["class", "confidence", "x1", "y1", "x2", "y2"]], hide_index=True, use_container_width=True)
 
 
if __name__ == "__main__":
    main()
