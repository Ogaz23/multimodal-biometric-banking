import cv2
import os
import pickle
import numpy as np
from iris_utils import extract_iris_code

SAVE_DIR = "iris_samples"
DB_FILE = "database/iris_database.pkl"
NUM_SAMPLES = 3

def capture_iris_samples(name, num_samples=NUM_SAMPLES):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    codes = []
    count = 0

    print(f"\nGet close to the camera with good lighting. We need {num_samples} clear eye captures.")
    print("Press SPACE to capture, ESC to cancel.\n")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"Sample {count+1}/{num_samples} - SPACE to capture",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Iris Enrollment", display)
        key = cv2.waitKey(1)

        if key % 256 == 27:
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return []
        elif key % 256 == 32:
            code, status = extract_iris_code(frame)
            if code is None:
                print(f"⚠️ {status} — try again, get closer with better lighting.")
                continue

            img_path = os.path.join(SAVE_DIR, f"{name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            codes.append(code)
            count += 1
            print(f"✅ Sample {count}/{num_samples} captured and processed.")

    cap.release()
    cv2.destroyAllWindows()
    return codes

def enroll_person(name):
    codes = capture_iris_samples(name)
    if not codes:
        print("No valid samples captured. Enrollment cancelled.")
        return

    # Store all codes (we'll compare against the best match, not average,
    # since binary codes don't average meaningfully)
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            database = pickle.load(f)
    else:
        database = {}

    database[name] = codes  # list of iris codes

    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)

    print(f"\n✅ {name} enrolled successfully with {len(codes)} iris sample(s)!")

if __name__ == "__main__":
    person_name = input("Enter the name of the person to enroll: ")
    enroll_person(person_name)