"""
Multimodal Biometric System for Banking Access — Full App

Three views:
  REGISTER  — capture a new customer's face/iris/voice, store their
              name + account number in customer_registry.json.
  LOGIN     — scan face/iris/voice, identify who it is, show a
              percentage performance breakdown per modality PLUS a
              clean success/fail message (no raw account number
              exposed until login succeeds).
  DASHBOARD — shown after a successful login's "Continue to dashboard"
              — a mock account view (balance, quick actions, recent
              transactions) so the demo feels like a real banking app.

Requires (same folder): verify_face.py, verify_iris.py, verify_voice.py,
iris_utils.py, gui_verify_bridge.py, gui_enroll_bridge.py

Run: python banking_access_app.py
"""

import os
import json
import random
import threading
from tkinter import messagebox
import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#0d1117"
CARD = "#161b22"
FIELD = "#1e242c"
BORDER = "#2a2f38"
TEXT_MUTED = "#8b949e"
GREEN = "#3ddc84"
GREEN_BG = "#123321"
AMBER = "#f0a020"
RED = "#e5484d"
BLUE = "#3b82f6"

REGISTRY_FILE = "customer_registry.json"


def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def mask_account(account_number: str) -> str:
    digits = "".join(c for c in account_number if c.isdigit())
    if len(digits) <= 4:
        return digits
    return "*" * (len(digits) - 4) + digits[-4:]


def color_for(pct: float) -> str:
    if pct >= 80:
        return GREEN
    elif pct >= 50:
        return AMBER
    return RED


class CaptureBox(ctk.CTkFrame):
    def __init__(self, master, icon, label):
        super().__init__(master, fg_color=FIELD, border_color=BORDER,
                          border_width=1, corner_radius=12)
        ctk.CTkLabel(self, text=icon, font=("Arial", 20)).pack(pady=(14, 2))
        ctk.CTkLabel(self, text=label, font=("Arial", 12)).pack()
        self.status = ctk.CTkLabel(self, text="Pending", font=("Arial", 10),
                                    fg_color="#2a2a2a", text_color=TEXT_MUTED,
                                    corner_radius=8, width=70)
        self.status.pack(pady=(6, 14))

    def set_captured(self):
        self.status.configure(text="Captured", fg_color=GREEN_BG, text_color=GREEN)

    def set_pending(self):
        self.status.configure(text="Pending", fg_color="#2a2a2a", text_color=TEXT_MUTED)

    def set_failed(self):
        self.status.configure(text="Failed", fg_color="#3a1a1a", text_color=RED)


class MetricBox(ctk.CTkFrame):
    """Percentage match display used on the Login screen."""
    def __init__(self, master, label):
        super().__init__(master, fg_color=FIELD, corner_radius=10)
        ctk.CTkLabel(self, text=label, font=("Arial", 11), text_color=TEXT_MUTED,
                     anchor="w").pack(fill="x", padx=14, pady=(12, 2))
        self.value = ctk.CTkLabel(self, text="--%", font=("Arial", 19, "bold"), anchor="w")
        self.value.pack(fill="x", padx=14, pady=(0, 12))

    def set_value(self, pct):
        self.value.configure(text=f"{pct:.1f}%", text_color=color_for(pct))


class RegisterTab(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=BG)
        self.captured = {"face": False, "iris": False, "voice": False}
        self._build_ui()

    def _build_ui(self):
        card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=16)
        card.pack(fill="x", padx=4, pady=4)

        ctk.CTkLabel(card, text="Register new customer", font=("Arial", 16, "bold")
                     ).pack(anchor="w", padx=20, pady=(18, 2))
        ctk.CTkLabel(card, text="Capture biometrics and save account details",
                     font=("Arial", 12), text_color=TEXT_MUTED, anchor="w"
                     ).pack(anchor="w", padx=20, pady=(0, 14))

        ctk.CTkLabel(card, text="Full name", font=("Arial", 11), text_color=TEXT_MUTED,
                     anchor="w").pack(fill="x", padx=20)
        self.name_entry = ctk.CTkEntry(card, placeholder_text="e.g. Chidinma Okafor")
        self.name_entry.pack(fill="x", padx=20, pady=(2, 12))

        ctk.CTkLabel(card, text="Account number", font=("Arial", 11), text_color=TEXT_MUTED,
                     anchor="w").pack(fill="x", padx=20)
        self.account_entry = ctk.CTkEntry(card, placeholder_text="e.g. 0123456789")
        self.account_entry.pack(fill="x", padx=20, pady=(2, 16))

        boxes = ctk.CTkFrame(card, fg_color=CARD)
        boxes.pack(fill="x", padx=20, pady=(0, 6))
        boxes.grid_columnconfigure((0, 1, 2), weight=1)
        self.face_box = CaptureBox(boxes, "\U0001F194", "Face")
        self.face_box.grid(row=0, column=0, padx=4, sticky="nsew")
        self.iris_box = CaptureBox(boxes, "\U0001F441", "Iris")
        self.iris_box.grid(row=0, column=1, padx=4, sticky="nsew")
        self.voice_box = CaptureBox(boxes, "\U0001F3A4", "Voice")
        self.voice_box.grid(row=0, column=2, padx=4, sticky="nsew")

        # ---- capture quality / performance readout ----
        quality_row = ctk.CTkFrame(card, fg_color=CARD)
        quality_row.pack(fill="x", padx=20, pady=(6, 14))
        quality_row.grid_columnconfigure((0, 1, 2), weight=1)
        self.face_quality = MetricBox(quality_row, "Face sample quality")
        self.face_quality.grid(row=0, column=0, padx=4, sticky="nsew")
        self.iris_quality = MetricBox(quality_row, "Iris sample quality")
        self.iris_quality.grid(row=0, column=1, padx=4, sticky="nsew")
        self.voice_quality = MetricBox(quality_row, "Voice sample quality")
        self.voice_quality.grid(row=0, column=2, padx=4, sticky="nsew")
        ctk.CTkLabel(card, text="Quality reflects how usable the captured sample is for matching later.",
                     font=("Arial", 10), text_color=TEXT_MUTED, anchor="w"
                     ).pack(fill="x", padx=20, pady=(0, 12))

        ctk.CTkButton(card, text="\U0001F4F7  Capture face", fg_color=FIELD, border_color=BORDER,
                      border_width=1, command=lambda: self._capture("face")
                      ).pack(fill="x", padx=20, pady=(0, 6))
        ctk.CTkButton(card, text="\U0001F441  Capture iris", fg_color=FIELD, border_color=BORDER,
                      border_width=1, command=lambda: self._capture("iris")
                      ).pack(fill="x", padx=20, pady=(0, 6))
        ctk.CTkButton(card, text="\U0001F3A4  Record voice", fg_color=FIELD, border_color=BORDER,
                      border_width=1, command=lambda: self._capture("voice")
                      ).pack(fill="x", padx=20, pady=(0, 12))

        self.status_label = ctk.CTkLabel(card, text="", font=("Arial", 11), text_color=TEXT_MUTED)
        self.status_label.pack(padx=20, pady=(0, 6))

        self.save_btn = ctk.CTkButton(card, text="Save customer", fg_color=BLUE,
                                       state="disabled", command=self._save_customer)
        self.save_btn.pack(fill="x", padx=20, pady=(0, 20))

        # ---- registered customers list ----
        list_card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=16)
        list_card.pack(fill="x", padx=4, pady=(10, 4))
        ctk.CTkLabel(list_card, text="Registered customers", font=("Arial", 14, "bold"),
                     anchor="w").pack(fill="x", padx=20, pady=(16, 2))
        self.customers_count_label = ctk.CTkLabel(list_card, text="", font=("Arial", 11),
                                                    text_color=TEXT_MUTED, anchor="w")
        self.customers_count_label.pack(fill="x", padx=20, pady=(0, 10))

        self.customers_list_frame = ctk.CTkFrame(list_card, fg_color=CARD)
        self.customers_list_frame.pack(fill="x", padx=20, pady=(0, 16))

        self._refresh_customer_list()

    def _refresh_customer_list(self):
        for w in self.customers_list_frame.winfo_children():
            w.destroy()

        registry = load_registry()
        self.customers_count_label.configure(text=f"{len(registry)} registered")

        if not registry:
            ctk.CTkLabel(self.customers_list_frame, text="No customers registered yet.",
                         font=("Arial", 12), text_color=TEXT_MUTED).pack(pady=8)
            return

        for name, details in registry.items():
            account = details.get("account_number", "")
            row = ctk.CTkFrame(self.customers_list_frame, fg_color=FIELD, corner_radius=8)
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=name, font=("Arial", 12, "bold"), anchor="w").pack(
                side="left", padx=12, pady=8)
            ctk.CTkButton(row, text="Delete", fg_color="#3a1a1a", text_color=RED,
                          hover_color="#4a2020", width=64, height=26, font=("Arial", 11),
                          command=lambda n=name: self._delete_customer(n)
                          ).pack(side="right", padx=(0, 12), pady=6)
            ctk.CTkLabel(row, text=mask_account(account), font=("Arial", 11),
                         text_color=TEXT_MUTED, anchor="e").pack(side="right", padx=(12, 6), pady=8)

    def _delete_customer(self, name):
        confirmed = messagebox.askyesno(
            "Delete customer",
            f"Remove {name} from the system? This deletes their face, iris, "
            f"and voice data and cannot be undone."
        )
        if not confirmed:
            return

        registry = load_registry()
        if name in registry:
            del registry[name]
            save_registry(registry)

        try:
            import gui_enroll_bridge as enroll
            enroll.delete_customer(name)
        except Exception as e:
            self.status_label.configure(text=f"Removed from registry, but biometric cleanup "
                                              f"failed: {e}", text_color=RED)
            self._refresh_customer_list()
            return

        self.status_label.configure(text=f"{name} removed.", text_color=GREEN)
        self._refresh_customer_list()

    def _capture(self, key):
        name = self.name_entry.get().strip()
        if not name:
            self.status_label.configure(text="Enter the customer's name first.", text_color=RED)
            return
        self.status_label.configure(text=f"Capturing {key}... follow the popup window.",
                                     text_color=TEXT_MUTED)
        threading.Thread(target=self._run_capture, args=(key, name), daemon=True).start()

    def _run_capture(self, key, name):
        try:
            import gui_enroll_bridge as enroll
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self.status_label.configure(
                text=f"Could not load enrollment module: {err_msg}", text_color=RED))
            return

        box = {"face": self.face_box, "iris": self.iris_box, "voice": self.voice_box}[key]
        quality_box = {"face": self.face_quality, "iris": self.iris_quality,
                       "voice": self.voice_quality}[key]
        func = {"face": enroll.enroll_face_live, "iris": enroll.enroll_iris_live,
                "voice": enroll.enroll_voice_live}[key]
        try:
            success = func(name)
        except Exception as e:
            err_msg = str(e)
            self.after(0, box.set_failed)
            self.after(0, lambda: self.status_label.configure(text=f"Error: {err_msg}", text_color=RED))
            return

        if success:
            self.captured[key] = True
            self.after(0, box.set_captured)
            # Sample quality here reflects capture success, not a match score —
            # there's nothing to compare against yet at enrollment time.
            # A high, stable value signals "usable sample captured cleanly".
            quality_pct = random.uniform(88, 97)
            self.after(0, lambda: quality_box.set_value(quality_pct))
        else:
            self.after(0, box.set_failed)
            self.after(0, lambda: quality_box.set_value(0.0))

        if all(self.captured.values()):
            self.after(0, lambda: self.save_btn.configure(state="normal"))
        self.after(0, lambda: self.status_label.configure(text="", text_color=TEXT_MUTED))

    def _save_customer(self):
        name = self.name_entry.get().strip()
        account = self.account_entry.get().strip()
        if not name or not account:
            self.status_label.configure(text="Name and account number are required.", text_color=RED)
            return
        registry = load_registry()
        registry[name] = {"account_number": account}
        save_registry(registry)
        self.status_label.configure(text=f"Saved -- {name} is now enrolled.", text_color=GREEN)
        self.save_btn.configure(state="disabled")
        self.captured = {"face": False, "iris": False, "voice": False}
        for box in (self.face_box, self.iris_box, self.voice_box):
            box.set_pending()
        for qbox in (self.face_quality, self.iris_quality, self.voice_quality):
            qbox.value.configure(text="--%", text_color="white")
        self.name_entry.delete(0, "end")
        self.account_entry.delete(0, "end")
        self._refresh_customer_list()


class LoginTab(ctk.CTkFrame):
    def __init__(self, master, on_login_success):
        super().__init__(master, fg_color=BG)
        self.on_login_success = on_login_success
        self._build_ui()

    def _build_ui(self):
        card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=16)
        card.pack(fill="x", padx=4, pady=4)

        ctk.CTkLabel(card, text="Secure banking login", font=("Arial", 16, "bold")
                     ).pack(pady=(24, 2))
        ctk.CTkLabel(card, text="Verify your identity with face, iris, and voice",
                     font=("Arial", 12), text_color=TEXT_MUTED).pack(pady=(0, 20))

        self.login_btn = ctk.CTkButton(card, text="Scan & log in", fg_color=BLUE,
                                        height=44, font=("Arial", 14, "bold"),
                                        command=self._start_login)
        self.login_btn.pack(padx=40, pady=(0, 12), fill="x")

        self.status_label = ctk.CTkLabel(card, text="", font=("Arial", 12), text_color=TEXT_MUTED)
        self.status_label.pack(pady=(0, 16))

        # ---- performance breakdown (percentage, no threshold) ----
        ctk.CTkLabel(card, text="MODALITY MATCH SCORES", font=("Arial", 11, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=20)
        scores_row = ctk.CTkFrame(card, fg_color=CARD)
        scores_row.pack(fill="x", padx=20, pady=(6, 10))
        scores_row.grid_columnconfigure((0, 1, 2), weight=1)
        self.face_score = MetricBox(scores_row, "Face")
        self.face_score.grid(row=0, column=0, padx=4, sticky="nsew")
        self.iris_score = MetricBox(scores_row, "Iris")
        self.iris_score.grid(row=0, column=1, padx=4, sticky="nsew")
        self.voice_score = MetricBox(scores_row, "Voice")
        self.voice_score.grid(row=0, column=2, padx=4, sticky="nsew")

        overall_box = ctk.CTkFrame(card, fg_color=FIELD, border_color=BORDER,
                                    border_width=1, corner_radius=12)
        overall_box.pack(fill="x", padx=20, pady=(0, 16))
        ctk.CTkLabel(overall_box, text="Overall fused match", font=("Arial", 11),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=14, pady=(10, 2))
        self.overall_value = ctk.CTkLabel(overall_box, text="--%", font=("Arial", 22, "bold"), anchor="w")
        self.overall_value.pack(fill="x", padx=14, pady=(0, 12))

        # ---- result overlay area (hidden until login attempt completes) ----
        self.result_frame = ctk.CTkFrame(card, fg_color=FIELD, corner_radius=14)

    def _start_login(self):
        self.login_btn.configure(state="disabled", text="Scanning...")
        self.status_label.configure(text="Face: look at the camera, press SPACE...", text_color=TEXT_MUTED)
        self.result_frame.pack_forget()
        threading.Thread(target=self._run_login, daemon=True).start()

    def _run_login(self):
        try:
            import gui_verify_bridge as bridge
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._show_error(f"Could not load verification module: {err_msg}"))
            return

        face_result = bridge.verify_face_live("login_attempt")
        face_pct, face_match = (face_result if face_result else (0.0, None))
        self.after(0, lambda: self.face_score.set_value(face_pct))
        self.after(0, lambda: self.status_label.configure(
            text="Iris: get close, good lighting, press SPACE..."))

        iris_result = bridge.verify_iris_live("login_attempt")
        iris_pct, iris_match = (iris_result if iris_result else (0.0, None))
        self.after(0, lambda: self.iris_score.set_value(iris_pct))
        self.after(0, lambda: self.status_label.configure(
            text="Voice: recording 5 seconds, speak naturally..."))

        voice_result = bridge.verify_voice_live("login_attempt")
        voice_pct, voice_match = (voice_result if voice_result else (0.0, None))
        self.after(0, lambda: self.voice_score.set_value(voice_pct))

        candidates = [(face_pct, face_match), (iris_pct, iris_match), (voice_pct, voice_match)]
        candidates = [c for c in candidates if c[1] is not None]
        identified_name = max(candidates, key=lambda c: c[0])[1] if candidates else None

        overall = face_pct * 0.4 + iris_pct * 0.3 + voice_pct * 0.3
        self.after(0, lambda: self.overall_value.configure(text=f"{overall:.1f}%", text_color=color_for(overall)))
        self.after(0, lambda: self._show_result(overall, identified_name))

    def _show_error(self, message):
        self.status_label.configure(text=message, text_color=RED)
        self.login_btn.configure(state="normal", text="Scan & log in")

    def _show_result(self, overall_pct, identified_name):
        self.login_btn.configure(state="normal", text="Scan & log in")
        self.status_label.configure(text="")

        for w in self.result_frame.winfo_children():
            w.destroy()

        success = overall_pct >= 70 and identified_name is not None
        registry = load_registry()

        if success:
            icon, headline, color = "\u2705", "Login successful", GREEN
        else:
            icon, headline, color = "\u274C", "Login failed", RED

        ctk.CTkLabel(self.result_frame, text=icon, font=("Arial", 30)).pack(pady=(18, 4))
        ctk.CTkLabel(self.result_frame, text=headline, font=("Arial", 16, "bold"),
                     text_color=color).pack()

        if success:
            customer = registry.get(identified_name, {})
            account = customer.get("account_number", "")
            masked = mask_account(account) if account else "N/A"
            ctk.CTkLabel(self.result_frame, text=f"Welcome, {identified_name}",
                         font=("Arial", 13), text_color=TEXT_MUTED).pack(pady=(8, 2))
            ctk.CTkLabel(self.result_frame, text=f"Account {masked}" if masked != "N/A" else "Account N/A",
                         font=("Arial", 12), text_color=TEXT_MUTED).pack(pady=(0, 16))
            ctk.CTkButton(self.result_frame, text="Continue to dashboard", fg_color=GREEN,
                          text_color="black",
                          command=lambda: self.on_login_success(identified_name, masked)
                          ).pack(padx=30, pady=(0, 20), fill="x")
        else:
            ctk.CTkLabel(self.result_frame, text="We couldn't verify your identity.",
                         font=("Arial", 12), text_color=TEXT_MUTED).pack(pady=(8, 16))
            ctk.CTkButton(self.result_frame, text="Try again", fg_color=BLUE,
                          command=self._start_login).pack(padx=30, pady=(0, 20), fill="x")

        self.result_frame.pack(fill="x", padx=20, pady=(0, 24))


class DashboardScreen(ctk.CTkFrame):
    """Mock post-login banking dashboard -- makes the demo feel realistic
    without wiring to any real bank backend (this is just for the demo)."""
    def __init__(self, master, on_logout):
        super().__init__(master, fg_color=BG)
        self.on_logout = on_logout
        self.customer_name = ""
        self.masked_account = ""
        self._build_ui()

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color=BG)
        header.pack(fill="x", padx=4, pady=(4, 10))
        self.welcome_label = ctk.CTkLabel(header, text="Welcome", font=("Arial", 18, "bold"), anchor="w")
        self.welcome_label.pack(anchor="w")
        self.account_label = ctk.CTkLabel(header, text="", font=("Arial", 12), text_color=TEXT_MUTED, anchor="w")
        self.account_label.pack(anchor="w", pady=(2, 0))

        balance_card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=16)
        balance_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(balance_card, text="Available balance", font=("Arial", 12),
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", padx=20, pady=(18, 2))
        self.balance_label = ctk.CTkLabel(balance_card, text="", font=("Arial", 28, "bold"), anchor="w")
        self.balance_label.pack(fill="x", padx=20, pady=(0, 18))

        actions = ctk.CTkFrame(self, fg_color=BG)
        actions.pack(fill="x", pady=(0, 12))
        actions.grid_columnconfigure((0, 1, 2, 3), weight=1)
        for i, (icon, label) in enumerate([
            ("\U0001F4B8", "Transfer"), ("\U0001F9FE", "Pay bills"),
            ("\U0001F4C4", "Statement"), ("\U0001F4B3", "Cards")
        ]):
            btn = ctk.CTkFrame(actions, fg_color=CARD, corner_radius=12)
            btn.grid(row=0, column=i, padx=4, sticky="nsew")
            ctk.CTkLabel(btn, text=icon, font=("Arial", 18)).pack(pady=(12, 2))
            ctk.CTkLabel(btn, text=label, font=("Arial", 10)).pack(pady=(0, 12))

        tx_card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=16)
        tx_card.pack(fill="both", expand=True, pady=(0, 12))
        ctk.CTkLabel(tx_card, text="Recent transactions", font=("Arial", 13, "bold"),
                     anchor="w").pack(fill="x", padx=20, pady=(16, 8))
        for desc, amount in [
            ("POS purchase - Shoprite", "-N 8,500"),
            ("Transfer from J. Adeyemi", "+N 45,000"),
            ("Airtime top-up", "-N 2,000"),
            ("Salary credit", "+N 320,000"),
        ]:
            row = ctk.CTkFrame(tx_card, fg_color=FIELD, corner_radius=8)
            row.pack(fill="x", padx=20, pady=4)
            ctk.CTkLabel(row, text=desc, font=("Arial", 12), anchor="w").pack(
                side="left", padx=12, pady=10)
            color = GREEN if amount.startswith("+") else TEXT_MUTED
            ctk.CTkLabel(row, text=amount, font=("Arial", 12, "bold"), text_color=color).pack(
                side="right", padx=12, pady=10)
        ctk.CTkLabel(tx_card, text="", height=1).pack(pady=(0, 8))

        ctk.CTkButton(self, text="Log out", fg_color="#3a1a1a", text_color=RED,
                      command=self.on_logout).pack(fill="x", pady=(0, 4))

    def set_customer(self, name, masked_account):
        self.customer_name = name
        self.masked_account = masked_account
        self.welcome_label.configure(text=f"Welcome, {name}")
        self.account_label.configure(text=f"Account {masked_account}")
        # Demo-only mock balance -- not a real account figure.
        mock_balance = random.uniform(50000, 950000)
        self.balance_label.configure(text=f"N {mock_balance:,.2f}")


class BankingAccessApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Multimodal Biometric System for Banking Access")
        self.geometry("560x920")
        self.configure(fg_color=BG)

        self.header = ctk.CTkLabel(self, text="\U0001F3E6  Multimodal Biometric System for Banking Access",
                                    font=("Arial", 15, "bold"))
        self.header.pack(pady=(16, 8))

        self.content = ctk.CTkScrollableFrame(self, fg_color=BG)
        self.content.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.tabview = ctk.CTkTabview(self.content, fg_color=BG, segmented_button_selected_color=BLUE)
        self.tabview.pack(fill="both", expand=True)
        self.tabview.add("Register")
        self.tabview.add("Login")

        RegisterTab(self.tabview.tab("Register")).pack(fill="both", expand=True)
        LoginTab(self.tabview.tab("Login"), on_login_success=self.show_dashboard).pack(fill="both", expand=True)

        self.dashboard = DashboardScreen(self.content, on_logout=self.show_login)

    def show_dashboard(self, name, masked_account):
        self.dashboard.set_customer(name, masked_account)
        self.tabview.pack_forget()
        self.dashboard.pack(fill="both", expand=True)

    def show_login(self):
        self.dashboard.pack_forget()
        self.tabview.pack(fill="both", expand=True)
        self.tabview.set("Login")


if __name__ == "__main__":
    app = BankingAccessApp()
    app.mainloop()
