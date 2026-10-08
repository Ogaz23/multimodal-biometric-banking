import sounddevice as sd
import numpy as np
import wave
import os
import pickle
import torch
from speechbrain.inference.speaker import EncoderClassifier

SAMPLE_RATE = 16000
DURATION = 5  # seconds per sample
NUM_SAMPLES = 3  # number of recordings per person
VOICE_DIR = "voice_samples"
DB_FILE = "database/voice_database.pkl"

print("Loading speaker recognition model... (this may take a moment the first time)")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
print("✅ Model loaded.\n")

def record_audio(filename, duration=DURATION):
    print(f"🎤 Recording for {duration} seconds... speak naturally (e.g. count 1-10 or say a sentence).")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished.")

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

from scipy.io import wavfile

def get_embedding(filename):
    fs, audio_data = wavfile.read(filename)
    audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
    # Normalize 16-bit PCM range (-32768 to 32767) to -1.0 to 1.0
    audio_tensor = audio_tensor / 32768.0
    embedding = classifier.encode_batch(audio_tensor)
    return embedding.squeeze().detach().numpy()

def enroll_person(name, num_samples=NUM_SAMPLES):
    embeddings = []

    for i in range(num_samples):
        input(f"\nPress ENTER to record sample {i+1}/{num_samples}...")
        filepath = os.path.join(VOICE_DIR, f"{name}_voice_{i}.wav")
        record_audio(filepath)
        embedding = get_embedding(filepath)
        embeddings.append(embedding)
        print(f"✅ Sample {i+1} processed.")

    avg_embedding = np.mean(embeddings, axis=0)

    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            database = pickle.load(f)
    else:
        database = {}

    database[name] = avg_embedding

    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)

    print(f"\n✅ {name} enrolled successfully with {num_samples} voice samples!")

if __name__ == "__main__":
    person_name = input("Enter the name of the person to enroll: ")
    enroll_person(person_name)