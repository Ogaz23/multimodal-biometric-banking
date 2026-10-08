
import cv2
import os
import pickle
import csv
import numpy as np
from datetime import datetime
from iris_utils import extract_iris_code, hamming_distance

DB_FILE = "database/iris_database.pkl"
LOG_FILE = "iris_verification_log.csv"
THRESHOLD = 0.35  # LOWER = better match for iris

def log_attempt(claimed_name, actual_identity, best_match, best_score, result):
    file_exists = os.path.exists(LOG_FILE)
    is_genuine = (claimed_name.strip().lower() == actual_identity.strip().lower())
    attempt_type = "genuine" if is_genuine else "impostor"
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "claimed_name", "actual_identity", "attempt_type",
                              "best_match", "hamming_distance", "result"])
        writer.writerow([datetime.now().isoformat(), claimed_name, actual_identity, attempt_type,
                          best_match, f"{best_score:.4f}", result])

def capture_live_iris():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("Get close, good lighting. Press SPACE to capture, ESC to cancel.")

    code = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Iris Verification - Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("Cancelled.")
            break
        elif key % 256 == 32:
            code, status = extract_iris_code(frame)
            if code is None:
                print(f"⚠️ {status} — try again.")
                continue
            print("✅ Iris captured.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return code

def verify_person(claimed_name, actual_identity, threshold=THRESHOLD, silent=False):
    if not os.path.exists(DB_FILE):
        print("❌ No enrolled irises found. Run enroll_iris.py first.")
        return None

    live_code = capture_live_iris()
    if live_code is None:
        return None

    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)

    best_match = None
    best_score = 1.0
    for name, stored_codes in database.items():
        person_best = min(hamming_distance(live_code, code) for code in stored_codes)
        if not silent:
            print(f"Hamming distance with {name}: {person_best:.4f}")
        if person_best < best_score:
            best_score = person_best
            best_match = name

    outcome = "ACCESS_GRANTED" if best_score <= threshold else "ACCESS_DENIED"
    print(f"{'✅' if outcome=='ACCESS_GRANTED' else '❌'} {outcome} — best match: {best_match} (distance: {best_score:.4f})")

    log_attempt(claimed_name, actual_identity, best_match, best_score, outcome)
    return best_score

def batch_test():
    print("=== BATCH TESTING MODE (Iris) ===\n")
    claimed_name = input("Claimed identity: ").strip()
    actual_identity = input("Actual identity (ground truth): ").strip()
    num_attempts = int(input("How many attempts to run? ").strip())

    is_genuine = claimed_name.lower() == actual_identity.lower()
    print(f"\nMode: {'GENUINE' if is_genuine else 'IMPOSTOR'} testing")
    print(f"Running {num_attempts} attempts.\n")

    scores = []
    for i in range(num_attempts):
        print(f"\n--- Attempt {i+1}/{num_attempts} ---")
        score = verify_person(claimed_name, actual_identity, silent=True)
        if score is not None:
            scores.append(score)

    if scores:
        print(f"\n=== BATCH COMPLETE ===")
        print(f"Attempts logged: {len(scores)}")
        print(f"Average distance: {np.mean(scores):.4f}")
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