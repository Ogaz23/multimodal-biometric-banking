import cv2
import os
import pickle
import csv
import numpy as np
from datetime import datetime
from deepface import DeepFace

DB_FILE = "database/face_database.pkl"
TEMP_IMG = "captured_faces/temp_verify.jpg"
LOG_FILE = "face_verification_log.csv"
THRESHOLD = 0.68

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def log_attempt(claimed_name, actual_identity, best_match, best_score, result):
    file_exists = os.path.exists(LOG_FILE)
    is_genuine = (claimed_name.strip().lower() == actual_identity.strip().lower())
    attempt_type = "genuine" if is_genuine else "impostor"
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "claimed_name", "actual_identity", "attempt_type",
                              "best_match", "score", "result"])
        writer.writerow([datetime.now().isoformat(), claimed_name, actual_identity, attempt_type,
                          best_match, f"{best_score:.4f}", result])

def capture_live_face():
    cap = cv2.VideoCapture(0)
    print("Look at the camera. Press SPACE to capture, ESC to cancel.")

    captured = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Verification - Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("Cancelled.")
            break
        elif key % 256 == 32:
            try:
                DeepFace.extract_faces(img_path=frame, enforce_detection=True)
            except ValueError:
                print("⚠️ No face detected — try again.")
                continue
            cv2.imwrite(TEMP_IMG, frame)
            captured = True
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured

def verify_person(claimed_name, actual_identity, threshold=THRESHOLD, silent=False):
    if not os.path.exists(DB_FILE):
        print("❌ No enrolled faces found. Run enroll_face.py first.")
        return None

    if not capture_live_face():
        return None

    result = DeepFace.represent(img_path=TEMP_IMG, model_name="ArcFace", enforce_detection=True)
    live_embedding = result[0]["embedding"]

    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)

    best_match = None
    best_score = -1
    for name, stored_embedding in database.items():
        score = cosine_similarity(live_embedding, stored_embedding)
        if not silent:
            print(f"Similarity with {name}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_match = name

    outcome = "ACCESS_GRANTED" if best_score >= threshold else "ACCESS_DENIED"
    print(f"{'✅' if outcome=='ACCESS_GRANTED' else '❌'} {outcome} — best match: {best_match} (score: {best_score:.4f})")

    log_attempt(claimed_name, actual_identity, best_match, best_score, outcome)
    return best_score

def batch_test():
    print("=== BATCH TESTING MODE (Face) ===\n")
    claimed_name = input("Claimed identity (who the system should think this is): ").strip()
    actual_identity = input("Actual identity (who is really in front of the camera): ").strip()
    num_attempts = int(input("How many attempts to run? ").strip())

    is_genuine = claimed_name.lower() == actual_identity.lower()
    print(f"\nMode: {'GENUINE' if is_genuine else 'IMPOSTOR'} testing")
    print(f"Running {num_attempts} attempts. Vary your angle/lighting slightly between captures.\n")

    scores = []
    for i in range(num_attempts):
        print(f"\n--- Attempt {i+1}/{num_attempts} ---")
        score = verify_person(claimed_name, actual_identity, silent=True)
        if score is not None:
            scores.append(score)

    if scores:
        print(f"\n=== BATCH COMPLETE ===")
        print(f"Attempts logged: {len(scores)}")
        print(f"Average score: {np.mean(scores):.4f}")
        print(f"Min: {np.min(scores):.4f} | Max: {np.max(scores):.4f}")
    print(f"📝 All results appended to {LOG_FILE}")

if __name__ == "__main__":
    print("1. Single verification")
    print("2. Batch testing (multiple attempts)")
    choice = input("Choose (1/2): ").strip()

    if choice == "2":
        batch_test()
    else:
        claimed = input("Who are you claiming to be?: ")
        actual = input("Who is actually in front of the camera (ground truth)?: ")
        verify_person(claimed, actual)