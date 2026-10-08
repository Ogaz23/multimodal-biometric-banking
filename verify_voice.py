
import sounddevice as sd
import numpy as np
import wave
import os
import pickle
import csv
import torch
from datetime import datetime
from scipy.io import wavfile
from speechbrain.inference.speaker import EncoderClassifier

SAMPLE_RATE = 16000
DURATION = 5
DB_FILE = "database/voice_database.pkl"
TEMP_WAV = "voice_samples/temp_verify.wav"
LOG_FILE = "voice_verification_log.csv"
THRESHOLD = 0.5

print("Loading speaker recognition model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
print("✅ Model loaded.\n")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def record_audio(filename, duration=DURATION):
    print(f"🎤 Recording for {duration} seconds... speak naturally.")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished.")
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

def get_embedding(filename):
    fs, audio_data = wavfile.read(filename)
    audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
    audio_tensor = audio_tensor / 32768.0
    embedding = classifier.encode_batch(audio_tensor)
    return embedding.squeeze().detach().numpy()

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

def verify_person(claimed_name, actual_identity, threshold=THRESHOLD, silent=False):
    if not os.path.exists(DB_FILE):
        print("❌ No enrolled voices found. Run enroll_voice.py first.")
        return None

    input("Press ENTER to record your verification sample...")
    record_audio(TEMP_WAV)

    live_embedding = get_embedding(TEMP_WAV)

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
    print("=== BATCH TESTING MODE (Voice) ===\n")
    claimed_name = input("Claimed identity: ").strip()
    actual_identity = input("Actual identity (ground truth): ").strip()
    num_attempts = int(input("How many attempts to run? ").strip())

    is_genuine = claimed_name.lower() == actual_identity.lower()
    print(f"\nMode: {'GENUINE' if is_genuine else 'IMPOSTOR'} testing")
    print(f"Running {num_attempts} attempts. Vary phrase/tone slightly between captures.\n")

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
        actual = input("Who is actually speaking (ground truth)?: ")
        verify_person(claimed, actual)