import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import pickle
import csv
import numpy as np
import torch
import threading
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

TEMP_FACE_IMG = "captured_faces/temp_dash.jpg"
TEMP_VOICE_WAV = "voice_samples/temp_dash.wav"
SAMPLE_RATE = 16000
VOICE_DURATION = 5

WEIGHT_FACE = 0.4
WEIGHT_VOICE = 0.3
WEIGHT_IRIS = 0.3
FUSION_THRESHOLD = 0.6
NUM_ENROLL_SAMPLES = 3

# ---------- COLOR TOKENS ----------
BG = "#0d1117"
PANEL = "#161b22"
PANEL_ALT = "#1c2128"
BORDER = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
ACCENT = "#3fb1ff"
SUCCESS = "#3fb950"
DANGER = "#f85149"
WARNING = "#d29922"

FONT_LABEL = ("Segoe UI", 10)
FONT_LABEL_BOLD = ("Segoe UI", 11, "bold")
FONT_HEADER = ("Segoe UI Semibold", 13, "bold")
FONT_MONO = ("Consolas", 11)
FONT_MONO_BOLD = ("Consolas", 13, "bold")
FONT_BIG = ("Consolas", 30, "bold")
FONT_TICKER = ("Consolas", 9)

print("Loading voice model... this may take a moment.")
voice_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
print("Voice model ready.")


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def normalize_iris_score(hamming_dist):
    similarity = 1 - (hamming_dist / 0.5)
    return max(0.0, min(1.0, similarity))


def get_voice_embedding(filename):
    fs, audio_data = wavfile.read(filename)
    audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0) / 32768.0
    embedding = voice_classifier.encode_batch(audio_tensor)
    return embedding.squeeze().detach().numpy()


def record_voice(filename, duration=VOICE_DURATION):
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


class ScoreMeter(tk.Frame):
    """A card-style instrument readout for one modality's similarity score."""
    def __init__(self, parent, label, sublabel, **kwargs):
        super().__init__(parent, bg=PANEL_ALT, highlightbackground=BORDER,
                          highlightthickness=1, **kwargs)
        self._last_score = None

        pad = tk.Frame(self, bg=PANEL_ALT)
        pad.pack(fill="x", padx=14, pady=12)

        header = tk.Frame(pad, bg=PANEL_ALT)
        header.pack(fill="x")
        text_block = tk.Frame(header, bg=PANEL_ALT)
        text_block.pack(side="left")
        tk.Label(text_block, text=label, fg=TEXT_PRIMARY, bg=PANEL_ALT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(text_block, text=sublabel, fg=TEXT_SECONDARY, bg=PANEL_ALT,
                 font=("Segoe UI", 8)).pack(anchor="w")

        self.value_lbl = tk.Label(header, text="—.———", fg=TEXT_SECONDARY, bg=PANEL_ALT,
                                   font=FONT_MONO_BOLD)
        self.value_lbl.pack(side="right")

        self.canvas = tk.Canvas(pad, height=8, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="x", pady=(10, 0))
        self.canvas.bind("<Configure>", lambda e: self._redraw(self._last_score))

    def set_score(self, score):
        self._last_score = score
        if score is None:
            self.value_lbl.configure(text="—.———", fg=TEXT_SECONDARY)
        else:
            self.value_lbl.configure(text=f"{score:.3f}", fg=TEXT_PRIMARY)
        self._redraw(score)

    def _redraw(self, score):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            return
        self.canvas.create_rectangle(0, 0, w, h, fill=BG, width=0)
        if score is None:
            return
        fill_w = max(2, int(w * min(score, 1.0)))
        color = SUCCESS if score >= 0.6 else (WARNING if score >= 0.4 else DANGER)
        self.canvas.create_rectangle(0, 0, fill_w, h, fill=color, width=0)
        tick_x = int(w * 0.6)
        self.canvas.create_line(tick_x, 0, tick_x, h, fill=TEXT_SECONDARY, width=1)


class StatusDot(tk.Canvas):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", PANEL)
        super().__init__(parent, width=10, height=10, highlightthickness=0, **kwargs)
        self.dot = self.create_oval(1, 1, 9, 9, fill=TEXT_SECONDARY, outline="")

    def set_color(self, color):
        self.itemconfig(self.dot, fill=color)


class BiometricDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Biometric Verification System")
        self.root.geometry("1280x830")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 720)

        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.video_running = True

        self._build_layout()
        self._update_video_feed()

    # ---------- LAYOUT ----------
    def _build_layout(self):
        self._build_topbar()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self._build_camera_panel(body)
        self._build_control_panel(body)
        self._build_log_panel()

    def _build_topbar(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=16, pady=16)

        title_block = tk.Frame(top, bg=BG)
        title_block.pack(side="left")

        name_row = tk.Frame(title_block, bg=BG)
        name_row.pack(anchor="w")
        mark = tk.Canvas(name_row, width=22, height=22, bg=BG, highlightthickness=0)
        mark.pack(side="left", padx=(0, 10))
        mark.create_oval(2, 2, 20, 20, outline=ACCENT, width=2)
        mark.create_oval(8, 8, 14, 14, fill=ACCENT, outline="")
        tk.Label(name_row, text="MULTIMODAL BIOMETRIC VERIFICATION",
                 fg=TEXT_PRIMARY, bg=BG, font=("Segoe UI Semibold", 16, "bold")).pack(side="left")

        tk.Label(title_block, text="Face   ·   Voice   ·   Iris   —   Score-Level Fusion Engine",
                 fg=TEXT_SECONDARY, bg=BG, font=("Segoe UI", 10)).pack(anchor="w", padx=(32, 0))

        status_block = tk.Frame(top, bg=BG)
        status_block.pack(side="right")

        self.clock_label = tk.Label(status_block, text="", fg=TEXT_SECONDARY, bg=BG,
                                     font=FONT_MONO)
        self.clock_label.pack(side="left", padx=(0, 24))
        self._tick_clock()

        self.status_dot = StatusDot(status_block, bg=BG)
        self.status_dot.pack(side="left", padx=(0, 8), pady=4)
        self.status_label = tk.Label(status_block, text="SYSTEM READY", fg=TEXT_SECONDARY, bg=BG,
                                      font=("Segoe UI", 10, "bold"))
        self.status_label.pack(side="left")
        self.status_dot.set_color(ACCENT)

        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill="x")

    def _panel(self, parent, **grid_kwargs):
        outer = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        outer.grid(**grid_kwargs)
        return outer

    def _build_camera_panel(self, parent):
        panel = self._panel(parent, row=0, column=0, sticky="nsew", padx=(0, 8), pady=(16, 0))
        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(inner, text="LIVE SENSOR FEED", fg=TEXT_SECONDARY, bg=PANEL,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 8))

        self.video_frame = tk.Frame(inner, bg="black", highlightbackground=BORDER, highlightthickness=1)
        self.video_frame.pack(fill="both", expand=True)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        name_row = tk.Frame(inner, bg=PANEL)
        name_row.pack(fill="x", pady=(16, 0))
        tk.Label(name_row, text="IDENTITY", fg=TEXT_SECONDARY, bg=PANEL,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.name_entry = tk.Entry(name_row, font=FONT_MONO, bg=PANEL_ALT, fg=TEXT_PRIMARY,
                                    insertbackground=TEXT_PRIMARY, relief="flat",
                                    highlightbackground=BORDER, highlightthickness=1)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=(10, 0), ipady=5)

        btn_row = tk.Frame(inner, bg=PANEL)
        btn_row.pack(fill="x", pady=(12, 0))
        self._make_button(btn_row, "ENROLL", self.start_enrollment, ACCENT).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        self._make_button(btn_row, "VERIFY", self.start_verification, SUCCESS).pack(
            side="left", fill="x", expand=True, padx=(6, 0))

    def _make_button(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, command=command, bg=color, fg="#0d1117",
                         activebackground=color, activeforeground="#0d1117",
                         font=("Segoe UI", 10, "bold"), relief="flat", bd=0, pady=11,
                         cursor="hand2")
        return btn

    def _build_control_panel(self, parent):
        panel = self._panel(parent, row=0, column=1, sticky="nsew", pady=(16, 0))
        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(inner, text="MODALITY SCORES", fg=TEXT_SECONDARY, bg=PANEL,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 10))

        self.face_meter = ScoreMeter(inner, "FACE", "ArcFace embedding · cosine similarity")
        self.face_meter.pack(fill="x", pady=(0, 8))
        self.voice_meter = ScoreMeter(inner, "VOICE", "ECAPA-TDNN · cosine similarity")
        self.voice_meter.pack(fill="x", pady=(0, 8))
        self.iris_meter = ScoreMeter(inner, "IRIS", "Daugman segmentation · Gabor code")
        self.iris_meter.pack(fill="x", pady=(0, 8))

        divider = tk.Frame(inner, bg=BORDER, height=1)
        divider.pack(fill="x", pady=(8, 16))

        tk.Label(inner, text="FUSION DECISION", fg=TEXT_SECONDARY, bg=PANEL,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 10))

        self.fusion_card = tk.Frame(inner, bg=PANEL_ALT, highlightbackground=BORDER,
                                     highlightthickness=1)
        self.fusion_card.pack(fill="x")
        card_inner = tk.Frame(self.fusion_card, bg=PANEL_ALT)
        card_inner.pack(fill="x", padx=18, pady=20)

        self.fusion_outcome_lbl = tk.Label(card_inner, text="AWAITING INPUT", fg=TEXT_SECONDARY,
                                            bg=PANEL_ALT, font=("Segoe UI", 12, "bold"))
        self.fusion_outcome_lbl.pack(anchor="w")
        self.fusion_score_lbl = tk.Label(card_inner, text="—.———", fg=TEXT_PRIMARY,
                                          bg=PANEL_ALT, font=FONT_BIG)
        self.fusion_score_lbl.pack(anchor="w", pady=(4, 0))
        self.fusion_thresh_lbl = tk.Label(card_inner, text=f"threshold {FUSION_THRESHOLD:.2f}",
                                           fg=TEXT_SECONDARY, bg=PANEL_ALT, font=FONT_MONO)
        self.fusion_thresh_lbl.pack(anchor="w", pady=(2, 0))

        weights_row = tk.Frame(inner, bg=PANEL)
        weights_row.pack(fill="x", pady=(16, 0))
        tk.Label(weights_row,
                 text=f"FUSION WEIGHTS   face {WEIGHT_FACE}   ·   voice {WEIGHT_VOICE}   ·   iris {WEIGHT_IRIS}",
                 fg=TEXT_SECONDARY, bg=PANEL, font=("Segoe UI", 8, "bold")).pack(anchor="w")

        spacer = tk.Frame(inner, bg=PANEL)
        spacer.pack(fill="both", expand=True)

        engine_card = tk.Frame(inner, bg=PANEL_ALT, highlightbackground=BORDER, highlightthickness=1)
        engine_card.pack(fill="x", pady=(16, 0))
        engine_inner = tk.Frame(engine_card, bg=PANEL_ALT)
        engine_inner.pack(fill="x", padx=14, pady=10)
        tk.Label(engine_inner, text="ENGINE", fg=TEXT_SECONDARY, bg=PANEL_ALT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(engine_inner, text="DeepFace · SpeechBrain · OpenCV", fg=TEXT_PRIMARY, bg=PANEL_ALT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

    def _build_log_panel(self):
        panel = tk.Frame(self.root, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        panel.pack(fill="x", padx=16, pady=16)
        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=16, pady=12)

        tk.Label(inner, text="VERIFICATION LOG", fg=TEXT_SECONDARY, bg=PANEL,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 8))

        list_frame = tk.Frame(inner, bg=PANEL)
        list_frame.pack(fill="x")

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.log_box = tk.Listbox(list_frame, height=7, bg=PANEL_ALT, fg=TEXT_PRIMARY,
                                   font=FONT_MONO, borderwidth=0, highlightthickness=0,
                                   selectbackground=BORDER, yscrollcommand=scrollbar.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.log_box.yview)

        self._load_history()

    def _load_history(self):
        if not os.path.exists(LOG_FILE):
            self.log_box.insert(tk.END, "  no attempts logged yet")
            return
        with open(LOG_FILE, "r") as f:
            reader = list(csv.DictReader(f))
        if not reader:
            self.log_box.insert(tk.END, "  no attempts logged yet")
        for row in reader[-20:]:
            self._add_log_line(row["timestamp"], row["claimed_name"], row["fused_score"], row["result"])

    def _add_log_line(self, timestamp, name, score, result):
        ts_short = timestamp.split("T")[1].split(".")[0] if "T" in timestamp else timestamp
        granted = "GRANTED" in result
        tag = "[ OK ]" if granted else "[DENY]"
        line = f"  {ts_short}   {tag}   {name:<14}  score={score}"
        self.log_box.insert(tk.END, line)
        self.log_box.itemconfig(tk.END, fg=SUCCESS if granted else DANGER)
        self.log_box.see(tk.END)

    # ---------- VIDEO FEED ----------
    def _update_video_feed(self):
        if self.video_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                fw = self.video_frame.winfo_width()
                fh = self.video_frame.winfo_height()
                if fw > 10 and fh > 10:
                    h, w = frame.shape[:2]
                    scale = min(fw / w, fh / h)
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    resized = cv2.resize(frame, (new_w, new_h))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                    self.video_label.configure(image=img)
                    self.video_label.image = img
        self.root.after(30, self._update_video_feed)

    # ---------- STATUS / CLOCK ----------
    def _tick_clock(self):
        self.clock_label.configure(text=datetime.now().strftime("%Y-%m-%d   %H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def set_status(self, text, color):
        self.root.after(0, lambda: (
            self.status_label.configure(text=text.upper(), fg=color),
            self.status_dot.set_color(color)
        ))

    # ---------- ENROLLMENT ----------
    def start_enrollment(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Enter a name before enrolling.")
            return
        self.set_status(f"Enrolling {name}", WARNING)
        threading.Thread(target=self._enroll_worker, args=(name,), daemon=True).start()

    def _enroll_worker(self, name):
        try:
            face_embeddings = []
            for i in range(NUM_ENROLL_SAMPLES):
                messagebox.showinfo("Face enrollment", f"Sample {i+1}/{NUM_ENROLL_SAMPLES} — look at the camera, then click OK.")
                frame = self.current_frame
                try:
                    DeepFace.extract_faces(img_path=frame, enforce_detection=True)
                except ValueError:
                    messagebox.showwarning("No face detected", "Skipping this sample.")
                    continue
                path = f"captured_faces/{name}_{i}.jpg"
                cv2.imwrite(path, frame)
                result = DeepFace.represent(img_path=path, model_name="ArcFace", enforce_detection=True)
                face_embeddings.append(result[0]["embedding"])

            if face_embeddings:
                avg_face = np.mean(face_embeddings, axis=0).tolist()
                db = pickle.load(open(FACE_DB, "rb")) if os.path.exists(FACE_DB) else {}
                db[name] = avg_face
                pickle.dump(db, open(FACE_DB, "wb"))

            voice_embeddings = []
            for i in range(NUM_ENROLL_SAMPLES):
                messagebox.showinfo("Voice enrollment", f"Sample {i+1}/{NUM_ENROLL_SAMPLES} — click OK, then speak for 5 seconds.")
                path = f"voice_samples/{name}_{i}.wav"
                record_voice(path)
                voice_embeddings.append(get_voice_embedding(path))

            if voice_embeddings:
                avg_voice = np.mean(voice_embeddings, axis=0)
                db = pickle.load(open(VOICE_DB, "rb")) if os.path.exists(VOICE_DB) else {}
                db[name] = avg_voice
                pickle.dump(db, open(VOICE_DB, "wb"))

            iris_codes = []
            for i in range(NUM_ENROLL_SAMPLES):
                messagebox.showinfo("Iris enrollment", f"Sample {i+1}/{NUM_ENROLL_SAMPLES} — get close, good lighting, click OK.")
                frame = self.current_frame
                code, status = extract_iris_code(frame)
                if code is None:
                    messagebox.showwarning("Iris capture failed", f"{status} — skipping this sample.")
                    continue
                iris_codes.append(code)

            if iris_codes:
                db = pickle.load(open(IRIS_DB, "rb")) if os.path.exists(IRIS_DB) else {}
                db[name] = iris_codes
                pickle.dump(db, open(IRIS_DB, "wb"))

            self.set_status("Enrollment complete", SUCCESS)
            messagebox.showinfo("Done", f"{name} enrolled across all available modalities.")

        except Exception as e:
            self.set_status("Enrollment failed", DANGER)
            messagebox.showerror("Error", str(e))

    # ---------- VERIFICATION ----------
    def start_verification(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Enter the claimed identity before verifying.")
            return
        self.set_status("Verifying", WARNING)
        threading.Thread(target=self._verify_worker, args=(name,), daemon=True).start()

    def _verify_worker(self, claimed_name):
        face_score = voice_score = iris_score = None

        try:
            if os.path.exists(FACE_DB):
                messagebox.showinfo("Face verification", "Click OK, then we capture your face.")
                frame = self.current_frame
                try:
                    DeepFace.extract_faces(img_path=frame, enforce_detection=True)
                    cv2.imwrite(TEMP_FACE_IMG, frame)
                    result = DeepFace.represent(img_path=TEMP_FACE_IMG, model_name="ArcFace", enforce_detection=True)
                    live_emb = result[0]["embedding"]
                    db = pickle.load(open(FACE_DB, "rb"))
                    face_score = max(cosine_similarity(live_emb, emb) for emb in db.values())
                except ValueError:
                    messagebox.showwarning("No face detected", "Skipping face modality.")

            self.root.after(0, lambda: self.face_meter.set_score(face_score))

            if os.path.exists(VOICE_DB):
                messagebox.showinfo("Voice verification", "Click OK, then speak for 5 seconds.")
                record_voice(TEMP_VOICE_WAV)
                live_emb = get_voice_embedding(TEMP_VOICE_WAV)
                db = pickle.load(open(VOICE_DB, "rb"))
                voice_score = max(cosine_similarity(live_emb, emb) for emb in db.values())

            self.root.after(0, lambda: self.voice_meter.set_score(voice_score))

            if os.path.exists(IRIS_DB):
                messagebox.showinfo("Iris verification", "Get close, good lighting, click OK to capture.")
                frame = self.current_frame
                code, status = extract_iris_code(frame)
                if code is not None:
                    db = pickle.load(open(IRIS_DB, "rb"))
                    best_dist = min(min(hamming_distance(code, c) for c in codes) for codes in db.values())
                    iris_score = normalize_iris_score(best_dist)
                else:
                    messagebox.showwarning("Iris capture failed", status)

            self.root.after(0, lambda: self.iris_meter.set_score(iris_score))

            scores = {}
            if face_score is not None: scores['face'] = (face_score, WEIGHT_FACE)
            if voice_score is not None: scores['voice'] = (voice_score, WEIGHT_VOICE)
            if iris_score is not None: scores['iris'] = (iris_score, WEIGHT_IRIS)

            if not scores:
                self.set_status("No modalities captured", DANGER)
                return

            total_w = sum(w for _, w in scores.values())
            fused = sum(s * w for s, w in scores.values()) / total_w
            outcome = "ACCESS_GRANTED" if fused >= FUSION_THRESHOLD else "ACCESS_DENIED"
            color = SUCCESS if outcome == "ACCESS_GRANTED" else DANGER

            def update_fusion_ui():
                self.fusion_outcome_lbl.configure(
                    text="ACCESS GRANTED" if outcome == "ACCESS_GRANTED" else "ACCESS DENIED",
                    fg=color)
                self.fusion_score_lbl.configure(text=f"{fused:.3f}")
                self.fusion_card.configure(highlightbackground=color)

            self.root.after(0, update_fusion_ui)
            self.set_status(outcome.replace("_", " "), color)

            self._log_fusion_attempt(claimed_name, face_score, voice_score, iris_score, fused, outcome)
            self.root.after(0, lambda: self._add_log_line(
                datetime.now().isoformat(), claimed_name, f"{fused:.4f}", outcome))

        except Exception as e:
            self.set_status("Verification failed", DANGER)
            messagebox.showerror("Error", str(e))

    def _log_fusion_attempt(self, claimed_name, face_score, voice_score, iris_score, fused_score, result):
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

    def on_close(self):
        self.video_running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometricDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()