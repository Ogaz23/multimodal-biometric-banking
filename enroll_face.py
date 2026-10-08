import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace

SAVE_DIR = "captured_faces"
DB_FILE = "database/face_database.pkl"
NUM_PHOTOS = 3  # number of photos per person for a more reliable embedding

def capture_faces(name, num_photos=NUM_PHOTOS):
    cap = cv2.VideoCapture(0)
    captured_paths = []
    count = 0

    print(f"\nWe'll capture {num_photos} photos. Slightly change angle/expression between each.")
    print("Press SPACE to capture, ESC to cancel.\n")

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"Photo {count+1}/{num_photos} - SPACE to capture",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enrollment", display)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return []
        elif key % 256 == 32:  # SPACE
            # Check a face is actually present before accepting the photo
            try:
                faces = DeepFace.extract_faces(img_path=frame, enforce_detection=True)
            except ValueError:
                print("⚠️ No face detected — try again, make sure your face is clearly visible.")
                continue

            img_path = os.path.join(SAVE_DIR, f"{name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            captured_paths.append(img_path)
            count += 1
            print(f"✅ Captured photo {count}/{num_photos}")

    cap.release()
    cv2.destroyAllWindows()
    return captured_paths

def enroll_person(name):
    img_paths = capture_faces(name)
    if not img_paths:
        print("No photos captured. Enrollment cancelled.")
        return

    print("\nExtracting face embeddings... please wait.")
    embeddings = []
    for path in img_paths:
        try:
            result = DeepFace.represent(img_path=path, model_name="ArcFace", enforce_detection=True)
            embeddings.append(result[0]["embedding"])
        except ValueError:
            print(f"⚠️ Skipped {path} — no face detected during embedding step.")

    if not embeddings:
        print("❌ No valid embeddings extracted. Enrollment failed.")
        return

    # Average the embeddings into one stable representation
    avg_embedding = np.mean(embeddings, axis=0).tolist()

    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            database = pickle.load(f)
    else:
        database = {}

    database[name] = avg_embedding

    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)

    print(f"\n✅ {name} enrolled successfully using {len(embeddings)} photo(s)!")

def list_enrolled():
    if not os.path.exists(DB_FILE):
        print("No one enrolled yet.")
        return
    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)
    print("\nEnrolled people:")
    for name in database:
        print(f" - {name}")

def delete_person(name):
    if not os.path.exists(DB_FILE):
        print("No database found.")
        return
    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)
    if name in database:
        del database[name]
        with open(DB_FILE, "wb") as f:
            pickle.dump(database, f)
        print(f"✅ {name} removed from database.")
    else:
        print(f"❌ {name} not found in database.")

if __name__ == "__main__":
    print("1. Enroll new person")
    print("2. List enrolled people")
    print("3. Delete a person")
    choice = input("Choose an option (1/2/3): ")

    if choice == "1":
        person_name = input("Enter the name of the person to enroll: ")
        enroll_person(person_name)
    elif choice == "2":
        list_enrolled()
    elif choice == "3":
        person_name = input("Enter the name of the person to delete: ")
        delete_person(person_name)
    else:
        print("Invalid choice.")