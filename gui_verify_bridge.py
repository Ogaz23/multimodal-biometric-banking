"""
gui_verify_bridge.py

Bridges your CLI-oriented verify_face.py / verify_iris.py / verify_voice.py
scripts into plain functions the GUI can call directly — no input()
prompts, no batch-testing menus. Each function does a LIVE capture
(webcam window for face/iris, 5s recording for voice) and returns a
0-100 percentage match against your enrolled database, using the SAME
models, embeddings, and database files as your original scripts.

Place this file in your biometric_system folder, alongside
verify_face.py, verify_iris.py, verify_voice.py, and iris_utils.py.

Each function returns (percentage, best_match_name) or None if
capture failed / nothing enrolled yet.
"""

import os
import pickle

# These imports run your existing scripts' module-level code once
# (loading the DeepFace/ArcFace and SpeechBrain models) — same as
# running them directly, just done once when the GUI starts.
import verify_face as vf
import verify_iris as vi
import verify_voice as vv


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def verify_face_live(claimed_name: str):
    """Opens the face capture window (SPACE to capture, ESC to cancel).
    Returns (percentage, best_match_name) or None."""
    if not os.path.exists(vf.DB_FILE):
        return None
    if not vf.capture_live_face():
        return None

    result = vf.DeepFace.represent(img_path=vf.TEMP_IMG, model_name="ArcFace",
                                    enforce_detection=True)
    live_embedding = result[0]["embedding"]

    with open(vf.DB_FILE, "rb") as f:
        database = pickle.load(f)

    best_score, best_match = -1, None
    for name, stored_embedding in database.items():
        score = vf.cosine_similarity(live_embedding, stored_embedding)
        if score > best_score:
            best_score, best_match = score, name

    outcome = "ACCESS_GRANTED" if best_score >= vf.THRESHOLD else "ACCESS_DENIED"
    vf.log_attempt(claimed_name, claimed_name, best_match, best_score, outcome)
    return _clamp(best_score * 100), best_match


def verify_iris_live(claimed_name: str):
    """Opens the iris capture window (SPACE to capture, ESC to cancel).
    Iris uses Hamming distance where LOWER = better match, so this
    converts to a percentage as (1 - distance) * 100.
    Returns (percentage, best_match_name) or None."""
    if not os.path.exists(vi.DB_FILE):
        return None
    live_code = vi.capture_live_iris()
    if live_code is None:
        return None

    with open(vi.DB_FILE, "rb") as f:
        database = pickle.load(f)

    best_score, best_match = 1.0, None
    for name, stored_codes in database.items():
        person_best = min(vi.hamming_distance(live_code, code) for code in stored_codes)
        if person_best < best_score:
            best_score, best_match = person_best, name

    outcome = "ACCESS_GRANTED" if best_score <= vi.THRESHOLD else "ACCESS_DENIED"
    vi.log_attempt(claimed_name, claimed_name, best_match, best_score, outcome)
    return _clamp((1 - best_score) * 100), best_match


def verify_voice_live(claimed_name: str, duration=None):
    """Records a live voice sample immediately (no 'press ENTER' prompt —
    the GUI button click is the trigger) and returns (percentage,
    best_match_name) or None."""
    if not os.path.exists(vv.DB_FILE):
        return None

    dur = duration or vv.DURATION
    vv.record_audio(vv.TEMP_WAV, duration=dur)
    live_embedding = vv.get_embedding(vv.TEMP_WAV)

    with open(vv.DB_FILE, "rb") as f:
        database = pickle.load(f)

    best_score, best_match = -1, None
    for name, stored_embedding in database.items():
        score = vv.cosine_similarity(live_embedding, stored_embedding)
        if score > best_score:
            best_score, best_match = score, name

    outcome = "ACCESS_GRANTED" if best_score >= vv.THRESHOLD else "ACCESS_DENIED"
    vv.log_attempt(claimed_name, claimed_name, best_match, best_score, outcome)
    return _clamp(best_score * 100), best_match
