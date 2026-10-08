import sounddevice as sd
import numpy as np
import wave

DURATION = 4  # seconds
SAMPLE_RATE = 16000  # SpeechBrain models expect 16kHz audio

def record_audio(filename, duration=DURATION):
    print(f"🎤 Recording for {duration} seconds... speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished.")

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"✅ Saved to {filename}")

if __name__ == "__main__":
    record_audio("voice_samples/mic_test.wav")