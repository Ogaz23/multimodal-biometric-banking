"""
Multimodal Biometric System for Banking Access — Enrollment Card

Styled to match the original enrollment-card mockup: light rounded card,
Face/Iris/Voice capture boxes with "Captured"/"Pending" status pills,
customer name/account fields, "Complete enrollment" button, and a
percentage-based system performance section below (no threshold).

Run: python enrollment_card_banking.py
"""

import customtkinter as ctk
import threading

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#1a1a1a"
CARD = "#232323"
FIELD = "#2c2c2c"
BORDER = "#3a3a3a"
TEXT_MUTED = "#9a9a9a"
GREEN_BG = "#173d26"
GREEN_FG = "#4ade80"
PENDING_BG = "#2f2f2f"
PENDING_FG = "#9a9a9a"
WHITE_BTN = "#ffffff"


class CaptureBox(ctk.CTkFrame):
    def __init__(self, master, icon, label):
        super().__init__(master, fg_color=FIELD, border_color=BORDER,
                          border_width=1, corner_radius=12)
        ctk.CTkLabel(self, text=icon, font=("Arial", 20)).pack(pady=(16, 4))
        ctk.CTkLabel(self, text=label, font=("Arial", 13)).pack()
        self.status = ctk.CTkLabel(self, text="Pending", font=("Arial", 11),
                                    fg_color=PENDING_BG, text_color=PENDING_FG,
                                    corner_radius=10, width=80, height=22)
        self.status.pack(pady=(8, 16))

    def set_captured(self):
        self.status.configure(text="Captured", fg_color=GREEN_BG, text_color=GREEN_FG)

    def set_pending(self):
        self.status.configure(text="Pending", fg_color=PENDING_BG, text_color=PENDING_FG)


class MetricBox(ctk.CTkFrame):
    def __init__(self, master, label):
        super().__init__(master, fg_color=FIELD, corner_radius=10)
        ctk.CTkLabel(self, text=label, font=("Arial", 11), text_color=TEXT_MUTED,
                     anchor="w").pack(fill="x", padx=14, pady=(12, 2))
        self.value = ctk.CTkLabel(self, text="--%", font=("Arial", 20, "bold"), anchor="w")
        self.value.pack(fill="x", padx=14, pady=(0, 12))

    def set_value(self, text, color="white"):
        self.value.configure(text=text, text_color=color)


class BankingEnrollmentCard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Multimodal Biometric System for Banking Access")
        self.geometry("520x950")
        self.configure(fg_color=BG)
        self.capture_state = {"face": False, "iris": False, "voice": False}
        self._build_ui()

    def _build_ui(self):
        outer = ctk.CTkScrollableFrame(self, fg_color=BG)
        outer.pack(fill="both", expand=True, padx=16, pady=16)

        card = ctk.CTkFrame(outer, fg_color=CARD, corner_radius=16)
        card.pack(fill="x")

        # ---- header ----
        ctk.CTkLabel(card, text="🏦", font=("Arial", 26)).pack(pady=(20, 4))
        ctk.CTkLabel(card, text="Multimodal Biometric System",
                     font=("Arial", 17, "bold")).pack()
        ctk.CTkLabel(card, text="for Banking Access", font=("Arial", 13),
                     text_color=TEXT_MUTED).pack(pady=(0, 18))

        # ---- customer details ----
        ctk.CTkLabel(card, text="Full name", font=("Arial", 11),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=24, pady=(0, 2))
        self.name_entry = ctk.CTkEntry(card, placeholder_text="Chidinma Okafor",
                                        fg_color=FIELD, border_color=BORDER)
        self.name_entry.pack(fill="x", padx=24, pady=(0, 12))

        ctk.CTkLabel(card, text="Account number", font=("Arial", 11),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=24, pady=(0, 2))
        self.account_entry = ctk.CTkEntry(card, placeholder_text="0123456789",
                                           fg_color=FIELD, border_color=BORDER)
        self.account_entry.pack(fill="x", padx=24, pady=(0, 18))

        # ---- capture boxes ----
        boxes_row = ctk.CTkFrame(card, fg_color=CARD)
        boxes_row.pack(fill="x", padx=24, pady=(0, 14))
        boxes_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.face_box = CaptureBox(boxes_row, "🆔", "Face")
        self.face_box.grid(row=0, column=0, padx=4, sticky="nsew")
        self.iris_box = CaptureBox(boxes_row, "👁", "Iris")
        self.iris_box.grid(row=0, column=1, padx=4, sticky="nsew")
        self.voice_box = CaptureBox(boxes_row, "🎤", "Voice")
        self.voice_box.grid(row=0, column=2, padx=4, sticky="nsew")

        # ---- capture / record button ----
        ctk.CTkButton(card, text="🎤  Record voice sample", fg_color=FIELD,
                      border_color=BORDER, border_width=1, text_color="white",
                      command=self._simulate_capture).pack(fill="x", padx=24, pady=(0, 10))

        self.complete_btn = ctk.CTkButton(
            card, text="Complete enrollment", fg_color=WHITE_BTN, text_color="black",
            state="disabled", command=self._complete_enrollment
        )
        self.complete_btn.pack(fill="x", padx=24, pady=(0, 10))

        self.verify_btn = ctk.CTkButton(
            card, text="Verify identity (live capture)", fg_color="#2563eb",
            command=self._on_verify_clicked
        )
        self.verify_btn.pack(fill="x", padx=24, pady=(0, 22))

        self.verify_status = ctk.CTkLabel(card, text="", font=("Arial", 11),
                                           text_color=TEXT_MUTED)
        self.verify_status.pack(padx=24, pady=(0, 4))

        ctk.CTkFrame(card, fg_color=BORDER, height=1).pack(fill="x", padx=24, pady=(0, 18))

        # ---- system performance (percentage-based, no threshold) ----
        ctk.CTkLabel(card, text="System performance", font=("Arial", 14, "bold"),
                     anchor="w").pack(fill="x", padx=24)
        ctk.CTkLabel(card, text="Percentage match per modality — no fixed threshold",
                     font=("Arial", 11), text_color=TEXT_MUTED, anchor="w").pack(
            fill="x", padx=24, pady=(2, 12))

        metrics_row = ctk.CTkFrame(card, fg_color=CARD)
        metrics_row.pack(fill="x", padx=24, pady=(0, 12))
        metrics_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.face_metric = MetricBox(metrics_row, "Face match")
        self.face_metric.grid(row=0, column=0, padx=4, sticky="nsew")
        self.iris_metric = MetricBox(metrics_row, "Iris match")
        self.iris_metric.grid(row=0, column=1, padx=4, sticky="nsew")
        self.voice_metric = MetricBox(metrics_row, "Voice match")
        self.voice_metric.grid(row=0, column=2, padx=4, sticky="nsew")

        overall_box = ctk.CTkFrame(card, fg_color=FIELD, border_color=BORDER,
                                    border_width=1, corner_radius=12)
        overall_box.pack(fill="x", padx=24, pady=(0, 24))
        ctk.CTkLabel(overall_box, text="Overall fused match", font=("Arial", 12),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=16, pady=(12, 2))
        self.overall_value = ctk.CTkLabel(overall_box, text="--%",
                                           font=("Arial", 24, "bold"), anchor="w")
        self.overall_value.pack(fill="x", padx=16, pady=(0, 14))

    # ---------- interaction ----------
    def _simulate_capture(self):
        """Placeholder — replace with real calls into your enroll_face.py /
        enroll_iris.py / enroll_voice.py scripts."""
        for key, box in [("face", self.face_box), ("iris", self.iris_box), ("voice", self.voice_box)]:
            self.capture_state[key] = True
            box.set_captured()
        if all(self.capture_state.values()):
            self.complete_btn.configure(state="normal")

    def _complete_enrollment(self):
        print("Enrolled:", self.name_entry.get(), self.account_entry.get())

    def _on_verify_clicked(self):
        name = self.name_entry.get().strip()
        if not name:
            self.verify_status.configure(text="Enter the customer's name first.", text_color="#e5484d")
            return

        self.verify_btn.configure(state="disabled", text="Verifying...")
        self.verify_status.configure(text="Starting verification — follow the popup windows.",
                                      text_color=TEXT_MUTED)
        threading.Thread(target=self._run_verification, args=(name,), daemon=True).start()

    def _run_verification(self, name: str):
        """Runs on a background thread so the window doesn't freeze during
        webcam capture / audio recording. Updates the GUI via self.after()."""
        try:
            import gui_verify_bridge as bridge
        except Exception as e:
            self.after(0, lambda: self.verify_status.configure(
                text=f"Could not load verification modules: {e}", text_color="#e5484d"))
            self.after(0, lambda: self.verify_btn.configure(state="normal", text="Verify identity (live capture)"))
            return

        results = {"face": None, "iris": None, "voice": None}

        self.after(0, lambda: self.verify_status.configure(text="Face: look at the camera, press SPACE..."))
        face_result = bridge.verify_face_live(name)
        results["face"] = face_result[0] if face_result else 0.0
        self.after(0, lambda: self.face_metric.set_value(
            f"{results['face']:.1f}%", self._color_for(results['face'])))

        self.after(0, lambda: self.verify_status.configure(text="Iris: get close, good lighting, press SPACE..."))
        iris_result = bridge.verify_iris_live(name)
        results["iris"] = iris_result[0] if iris_result else 0.0
        self.after(0, lambda: self.iris_metric.set_value(
            f"{results['iris']:.1f}%", self._color_for(results['iris'])))

        self.after(0, lambda: self.verify_status.configure(text="Voice: recording 5 seconds, speak naturally..."))
        voice_result = bridge.verify_voice_live(name)
        results["voice"] = voice_result[0] if voice_result else 0.0
        self.after(0, lambda: self.voice_metric.set_value(
            f"{results['voice']:.1f}%", self._color_for(results['voice'])))

        overall = (results["face"] * 0.4 + results["iris"] * 0.3 + results["voice"] * 0.3)
        self.after(0, lambda: self.overall_value.configure(
            text=f"{overall:.1f}%", text_color=self._color_for(overall)))
        self.after(0, lambda: self.verify_status.configure(
            text="Verification complete.", text_color=GREEN_FG))
        self.after(0, lambda: self.verify_btn.configure(
            state="normal", text="Verify identity (live capture)"))

    def _color_for(self, v):
        if v >= 80:
            return GREEN_FG
        elif v >= 50:
            return "#f0a020"
        return "#e5484d"

    def set_scores(self, face: float, iris: float, voice: float,
                   weights=(0.4, 0.3, 0.3)):
        """Call this with REAL percentage scores from your verify_*.py scripts.
        No accept/reject threshold — just displays the percentages."""
        def color_for(v):
            if v >= 80:
                return GREEN_FG
            elif v >= 50:
                return "#f0a020"
            return "#e5484d"

        self.face_metric.set_value(f"{face:.1f}%", color_for(face))
        self.iris_metric.set_value(f"{iris:.1f}%", color_for(iris))
        self.voice_metric.set_value(f"{voice:.1f}%", color_for(voice))

        overall = face * weights[0] + iris * weights[1] + voice * weights[2]
        self.overall_value.configure(text=f"{overall:.1f}%", text_color=color_for(overall))


if __name__ == "__main__":
    app = BankingEnrollmentCard()
    app.mainloop()
