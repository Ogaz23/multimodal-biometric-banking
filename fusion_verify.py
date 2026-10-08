import cv2
import os
import pickle
import csv
import numpy as np
import torch
from datetime import datetime
from scipy.io import wavfile
import sounddevice as sd
import wave
from deepface import DeepFace
from speechbrain.inference.speaker import EncoderClassifier
from iris_utils import extract_iris_code, hamming_distance

# ---------- CONFIG ----------
FACE_DB = "database/face_database.pkl"
VOICE_DB = "database/voice_database.pkl"
IRIS_DB = "database/iris_database.pkl"
LOG_FILE = "fusion_verification_log.csv"

TEMP_FACE_IMG = "captured_faces/temp_fusion.jpg"
TEMP_VOICE_WAV = "voice_samples/temp_fusion.wav"
SAMPLE_RATE = 16000
VOICE_DURATION = 5

# Weights for each modality (must sum to 1.0) - tune these based on individual accuracy
WEIGHT_FACE = 0.4
WEIGHT_VOICE = 0.3
WEIGHT_IRIS = 0.3

FUSION_THRESHOLD = 0.6  # final combined score threshold (0-1 scale, higher = better)

print("Loading voice model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
print("✅ Voice model loaded.\n")

# ---------- HELPER FUNCTIONS ----------

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalize_iris_score(hamming_dist):
    """Convert Hamming distance (lower=better) to a 0-1 similarity score (higher=better)."""
    # Hamming distance ranges roughly 0 (identical) to 0.5 (random)
    # Convert: similarity = 1 - (distance / 0.5), clipped to [0, 1]
    similarity = 1 - (hamming_dist / 0.5)
    return max(0.0, min(1.0, similarity))

def log_attempt(claimed_name, face_score, voice_score, iris_score, fused_score, result):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "claimed_name", "face_score", "voice_score",
                              "iris_score", "fused_score", "result"])
        writer.writerow([datetime.now().isoformat(), claimed_name,
                          f"{face_score:.4f}" if face_score is not None else "N/A",
                          f"{voice_score:.4f}" if voice_score is not None else "N/A",
                          f"{iris_score:.4f}" if iris_score is not None else "N/A",
                          f"{fused_score:.4f}", result])

# ---------- FACE ----------

def get_face_score():
    if not os.path.exists(FACE_DB):
        print("⚠️ No face database found — skipping face.")
        return None

    cap = cv2.VideoCapture(0)
    print("\n[FACE] Press SPACE to capture, ESC to skip.")
    img_captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Face Verification - SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:
            break
        elif key % 256 == 32:
            try:
                DeepFace.extract_faces(img_path=frame, enforce_detection=True)
            except ValueError:
                print("⚠️ No face detected, try again.")
                continue
            cv2.imwrite(TEMP_FACE_IMG, frame)
            img_captured = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if not img_captured:
        return None

    result = DeepFace.represent(img_path=TEMP_FACE_IMG, model_name="ArcFace", enforce_detection=True)
    live_embedding = result[0]["embedding"]

    with open(FACE_DB, "rb") as f:
        database = pickle.load(f)

    best_score = max(cosine_similarity(live_embedding, emb) for emb in database.values())
    print(f"[FACE] Best similarity: {best_score:.4f}")
    return best_score

# ---------- VOICE ----------

def get_voice_score():
    if not os.path.exists(VOICE_DB):
        print("⚠️ No voice database found — skipping voice.")
        return None

    input("\n[VOICE] Press ENTER to record...")
    print(f"🎤 Recording for {VOICE_DURATION} seconds...")
    audio = sd.rec(int(VOICE_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished.")

    with wave.open(TEMP_VOICE_WAV, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    fs, audio_data = wavfile.read(TEMP_VOICE_WAV)
    audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0) / 32768.0
    live_embedding = classifier.encode_batch(audio_tensor).squeeze().detach().numpy()

    with open(VOICE_DB, "rb") as f:
        database = pickle.load(f)

    best_score = max(cosine_similarity(live_embedding, emb) for emb in database.values())
    print(f"[VOICE] Best similarity: {best_score:.4f}")
    return best_score

# ---------- IRIS ----------

def get_iris_score():
    if not os.path.exists(IRIS_DB):
        print("⚠️ No iris database found — skipping iris.")
        return None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("\n[IRIS] Get close, good lighting. Press SPACE to capture, ESC to skip.")

    code = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Iris Verification - SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:
            break
        elif key % 256 == 32:
            code, status = extract_iris_code(frame)
            if code is None:
                print(f"⚠️ {status}, try again.")
                continue
            break

    cap.release()
    cv2.destroyAllWindows()

    if code is None:
        return None

    with open(IRIS_DB, "rb") as f:
        database = pickle.load(f)

    best_distance = min(
        min(hamming_distance(code, stored) for stored in stored_codes)
        for stored_codes in database.values()
    )
    similarity = normalize_iris_score(best_distance)
    print(f"[IRIS] Hamming distance: {best_distance:.4f} -> normalized similarity: {similarity:.4f}")
    return similarity

# ---------- FUSION ----------

def fused_verify(claimed_name="unknown"):
    print("=" * 50)
    print("MULTIMODAL BIOMETRIC VERIFICATION")
    print("=" * 50)

    face_score = get_face_score()
    voice_score = get_voice_score()
    iris_score = get_iris_score()

    # Only fuse the modalities that were actually captured, re-normalizing weights
    scores = {}
    if face_score is not None:
        scores['face'] = (face_score, WEIGHT_FACE)
    if voice_score is not None:
        scores['voice'] = (voice_score, WEIGHT_VOICE)
    if iris_score is not None:
        scores['iris'] = (iris_score, WEIGHT_IRIS)

    if not scores:
        print("❌ No modalities captured. Cannot verify.")
        return

    total_weight = sum(w for _, w in scores.values())
    fused_score = sum(s * w for s, w in scores.values()) / total_weight

    print("\n" + "=" * 50)
    print("FUSION RESULT")
    print("=" * 50)
    for name, (score, weight) in scores.items():
        print(f"  {name.capitalize()}: {score:.4f} (weight: {weight})")
    print(f"\n  Fused score: {fused_score:.4f}  (threshold: {FUSION_THRESHOLD})")

    if fused_score >= FUSION_THRESHOLD:
        outcome = "ACCESS_GRANTED"
        print(f"\n✅ ACCESS GRANTED")
    else:
        outcome = "ACCESS_DENIED"
        print(f"\n❌ ACCESS DENIED")

    log_attempt(claimed_name, face_score, voice_score, iris_score, fused_score, outcome)
    print(f"📝 Logged to {LOG_FILE}")

if __name__ == "__main__":
    claimed = input("Who are you claiming to be? (for logging): ")
    fused_verify(claimed_name=claimed)