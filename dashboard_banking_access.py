"""
Multimodal Biometric System for Banking Access — Dashboard

Redesigned dashboard: instead of a pass/fail threshold decision, this
shows a PERCENTAGE MATCH SCORE for each modality (face, iris, voice)
plus an overall fused percentage match. A customer details section
(name + account number) frames this as a banking access screen.

Run: python dashboard_banking_access.py
"""

import customtkinter as ctk
from datetime import datetime

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#2a2f38"
TEXT_MUTED = "#8b949e"
TEXT_DIM = "#6e7681"
BLUE = "#3b82f6"
GREEN = "#3ddc84"
AMBER = "#f0a020"


class ScoreCard(ctk.CTkFrame):
    """One modality's percentage match score (face / iris / voice)."""

    def __init__(self, master, title, subtitle):
        super().__init__(master, fg_color=PANEL, border_color=BORDER,
                          border_width=1, corner_radius=10)
        ctk.CTkLabel(self, text=title, font=("Arial", 13, "bold"),
                     anchor="w").pack(fill="x", padx=14, pady=(12, 0))
        ctk.CTkLabel(self, text=subtitle, font=("Arial", 11),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=14, pady=(0, 8))

        self.pct_label = ctk.CTkLabel(self, text="--%", font=("Arial", 22, "bold"))
        self.pct_label.pack(anchor="w", padx=14)

        self.bar = ctk.CTkProgressBar(self, height=6, progress_color=BLUE)
        self.bar.pack(fill="x", padx=14, pady=(6, 14))
        self.bar.set(0)

    def set_score(self, percent: float):
        """percent: 0-100. Colors the bar/label by rough confidence band."""
        self.pct_label.configure(text=f"{percent:.1f}%")
        self.bar.set(percent / 100)
        if percent >= 80:
            color = GREEN
        elif percent >= 50:
            color = AMBER
        else:
            color = "#e5484d"
        self.pct_label.configure(text_color=color)
        self.bar.configure(progress_color=color)


class BankingAccessDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Multimodal Biometric System for Banking Access")
        self.geometry("980x680")
        self.configure(fg_color=BG)
        self._build_ui()

    def _build_ui(self):
        # ---- Header ----
        header = ctk.CTkFrame(self, fg_color=BG)
        header.pack(fill="x", padx=24, pady=(20, 10))

        title_row = ctk.CTkFrame(header, fg_color=BG)
        title_row.pack(fill="x")
        ctk.CTkLabel(title_row, text="🏦  MULTIMODAL BIOMETRIC SYSTEM FOR BANKING ACCESS",
                     font=("Arial", 18, "bold")).pack(side="left")
        self.clock_label = ctk.CTkLabel(title_row, text="", font=("Arial", 12),
                                         text_color=TEXT_MUTED)
        self.clock_label.pack(side="right")
        self._tick_clock()

        ctk.CTkLabel(header, text="Face · Iris · Voice — Percentage-Based Match Scoring",
                     font=("Arial", 12), text_color=TEXT_MUTED, anchor="w").pack(fill="x", pady=(2, 0))

        # ---- Body: left (customer + video feed placeholder) / right (scores) ----
        body = ctk.CTkFrame(self, fg_color=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(10, 20))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        # ---- Left: customer details + capture area ----
        left = ctk.CTkFrame(body, fg_color=PANEL, border_color=BORDER,
                             border_width=1, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(left, text="CUSTOMER DETAILS", font=("Arial", 12, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=16, pady=(16, 8))

        form = ctk.CTkFrame(left, fg_color=BG, border_color=BORDER,
                             border_width=1, corner_radius=8)
        form.pack(fill="x", padx=16, pady=(0, 16))

        row1 = ctk.CTkFrame(form, fg_color="transparent")
        row1.pack(fill="x", padx=14, pady=(14, 8))
        ctk.CTkLabel(row1, text="Full name", font=("Arial", 11),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x")
        self.name_entry = ctk.CTkEntry(row1, placeholder_text="e.g. Chidinma Okafor")
        self.name_entry.pack(fill="x", pady=(4, 0))

        row2 = ctk.CTkFrame(form, fg_color="transparent")
        row2.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkLabel(row2, text="Account number", font=("Arial", 11),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x")
        self.account_entry = ctk.CTkEntry(row2, placeholder_text="e.g. 0123456789")
        self.account_entry.pack(fill="x", pady=(4, 0))

        ctk.CTkLabel(left, text="LIVE SENSOR FEED", font=("Arial", 12, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=16, pady=(4, 8))

        self.feed_placeholder = ctk.CTkFrame(left, fg_color="#000000", border_color=BORDER,
                                              border_width=1, corner_radius=8, height=280)
        self.feed_placeholder.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkLabel(self.feed_placeholder, text="camera feed renders here\n(wire to OpenCV capture)",
                     text_color=TEXT_DIM, font=("Arial", 12)).place(relx=0.5, rely=0.5, anchor="center")

        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkButton(btn_row, text="ENROLL", fg_color=BLUE, height=38,
                      command=self._on_enroll).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(btn_row, text="VERIFY", fg_color=GREEN, text_color="black", height=38,
                      command=self._on_verify).pack(side="left", expand=True, fill="x", padx=(6, 0))

        # ---- Right: modality scores + overall match ----
        right = ctk.CTkFrame(body, fg_color=BG)
        right.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(right, text="MODALITY MATCH SCORES", font=("Arial", 12, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", pady=(0, 8))

        self.face_score = ScoreCard(right, "FACE", "ArcFace embedding · cosine similarity")
        self.face_score.pack(fill="x", pady=(0, 10))
        self.iris_score = ScoreCard(right, "IRIS", "Daugman segmentation · Gabor code")
        self.iris_score.pack(fill="x", pady=(0, 10))
        self.voice_score = ScoreCard(right, "VOICE", "ECAPA-TDNN · cosine similarity")
        self.voice_score.pack(fill="x", pady=(0, 14))

        ctk.CTkFrame(right, fg_color=BORDER, height=1).pack(fill="x", pady=(0, 14))

        ctk.CTkLabel(right, text="OVERALL MATCH (FUSED)", font=("Arial", 12, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", pady=(0, 8))

        overall_panel = ctk.CTkFrame(right, fg_color=PANEL, border_color=BORDER,
                                      border_width=1, corner_radius=10)
        overall_panel.pack(fill="x")
        self.overall_label = ctk.CTkLabel(overall_panel, text="--%",
                                           font=("Arial", 32, "bold"))
        self.overall_label.pack(pady=(18, 4))
        self.overall_status = ctk.CTkLabel(overall_panel, text="AWAITING CAPTURE",
                                            font=("Arial", 12), text_color=TEXT_MUTED)
        self.overall_status.pack(pady=(0, 18))

        ctk.CTkLabel(right, text="Fusion weights — Face 0.4 · Voice 0.3 · Iris 0.3",
                     font=("Arial", 10), text_color=TEXT_DIM).pack(pady=(10, 0), anchor="w")

    def _tick_clock(self):
        self.clock_label.configure(text=datetime.now().strftime("%Y-%m-%d   %H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ---------- wiring hooks ----------
    def _on_enroll(self):
        """Hook: call your enroll_face.py / enroll_iris.py / enroll_voice.py here,
        using self.name_entry.get() and self.account_entry.get() as the customer ID."""
        print("Enroll pressed for:", self.name_entry.get(), self.account_entry.get())

    def _on_verify(self):
        """Hook: call your verify_face.py / verify_iris.py / verify_voice.py +
        fusion_verify.py here, then feed the results into set_scores()."""
        print("Verify pressed")
        # Preview-only placeholder values so you can see the layout live.
        # Replace this call with real scores from your verification scripts.
        self.set_scores(face=91.2, iris=88.7, voice=76.4)

    def set_scores(self, face: float, iris: float, voice: float,
                   weights=(0.4, 0.3, 0.3)):
        """Call this with REAL percentage match scores (0-100) from your
        verify_face.py / verify_iris.py / verify_voice.py outputs.
        No threshold accept/reject — just displays the percentages, per
        your supervisor's feedback."""
        self.face_score.set_score(face)
        self.iris_score.set_score(iris)
        self.voice_score.set_score(voice)

        w_face, w_voice, w_iris = weights[0], weights[2], weights[1]
        # NOTE: weights order documented above the label is Face/Voice/Iris —
        # keep this consistent with whatever fusion_verify.py actually uses.
        overall = face * weights[0] + iris * weights[1] + voice * weights[2]
        self.overall_label.configure(text=f"{overall:.1f}%")
        if overall >= 80:
            self.overall_label.configure(text_color=GREEN)
            self.overall_status.configure(text="LIKELY MATCH")
        elif overall >= 50:
            self.overall_label.configure(text_color=AMBER)
            self.overall_status.configure(text="UNCERTAIN — REVIEW")
        else:
            self.overall_label.configure(text_color="#e5484d")
            self.overall_status.configure(text="LIKELY NO MATCH")


if __name__ == "__main__":
    app = BankingAccessDashboard()
    app.mainloop()
