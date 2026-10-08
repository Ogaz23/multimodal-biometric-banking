# multimodal-biometric-banking
Multimodal biometric authentication for banking access using face, iris, and voice recognition with CCA-based fusion (Python)
# Multimodal Biometric System for Banking Access

A multimodal biometric authentication system for banking access that combines face, iris, and voice recognition, with CCA-based (Canonical Correlation Analysis) feature fusion. Built in Python as my final-year project.

## Features
- Face, iris, and voice enrollment and verification
- Fusion of the three modalities using CCA
- Percentage-based match scores instead of a simple accept/reject
- Banking access GUI built with CustomTkinter, with a mock post-login dashboard
- Evaluation script for performance metrics

## Tech used
- Python
- DeepFace (ArcFace) for face recognition
- Custom Daugman-based iris recognition
- SpeechBrain (ECAPA-TDNN) for voice recognition
- OpenCV
- CustomTkinter

## Project structure
- `enroll_face.py`, `enroll_iris.py`, `enroll_voice.py`: enroll a user for each modality
- `verify_face.py`, `verify_iris.py`, `verify_voice.py`: verify each modality
- `fusion_verify.py`: combines the modalities
- `banking_access_app.py`: main banking access GUI
- `evaluate_metrics.py`: performance evaluation

## How to run
1. Install Python 3.x
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python banking_access_app.py`

Note: biometric samples and pretrained models are not included in this repo for privacy and size reasons.

## Results
Results coming soon.

## Author
Ogazie
