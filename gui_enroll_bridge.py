"""
gui_enroll_bridge.py

Enrollment counterpart to gui_verify_bridge.py. Captures live face/iris/
voice and SAVES them into the same database files your verify_*.py
scripts read from — so a customer registered here can be verified later.

Database formats (matched to what verify_face.py / verify_iris.py /
verify_voice.py expect):
  - face_database.pkl  : { name: single_embedding }
  - voice_database.pkl : { name: single_embedding }
  - iris_database.pkl  : { name: [code1, code2, ...] }  (list, since
    verify_iris.py takes the MIN distance across all stored codes)

Place this file in your biometric_system folder.
"""

import os
import pickle

import verify_face as vf
import verify_iris as vi
import verify_voice as vv


def _load_db(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_db(path, db):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(db, f)


def enroll_face_live(name: str) -> bool:
    """Opens the face capture window (SPACE to capture, ESC to cancel).
    Returns True on success."""
    if not vf.capture_live_face():
        return False
    result = vf.DeepFace.represent(img_path=vf.TEMP_IMG, model_name="ArcFace",
                                    enforce_detection=True)
    embedding = result[0]["embedding"]
    db = _load_db(vf.DB_FILE)
    db[name] = embedding
    _save_db(vf.DB_FILE, db)
    return True


def enroll_iris_live(name: str) -> bool:
    """Opens the iris capture window (SPACE to capture, ESC to cancel).
    Appends this code to the customer's list of stored codes (so multiple
    enrollments improve matching). Returns True on success."""
    code = vi.capture_live_iris()
    if code is None:
        return False
    db = _load_db(vi.DB_FILE)
    db.setdefault(name, [])
    db[name].append(code)
    _save_db(vi.DB_FILE, db)
    return True


def enroll_voice_live(name: str, duration=None) -> bool:
    """Records a live voice sample immediately (no 'press ENTER' prompt).
    Returns True on success."""
    dur = duration or vv.DURATION
    vv.record_audio(vv.TEMP_WAV, duration=dur)
    embedding = vv.get_embedding(vv.TEMP_WAV)
    db = _load_db(vv.DB_FILE)
    db[name] = embedding
    _save_db(vv.DB_FILE, db)
    return True


def delete_customer(name: str):
    """Removes this customer's enrolled biometric data from the face,
    iris, and voice databases. Safe to call even if the name is missing
    from one or more databases."""
    for db_file in (vf.DB_FILE, vv.DB_FILE, vi.DB_FILE):
        db = _load_db(db_file)
        if name in db:
            del db[name]
            _save_db(db_file, db)
