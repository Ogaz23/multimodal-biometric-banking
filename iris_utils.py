import cv2
import numpy as np

def detect_eye_region(frame):
    """Detect eye region using Haar cascade, return cropped eye image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(60, 60))

    if len(eyes) == 0:
        return None

    # Take the largest detected eye region
    (x, y, w, h) = max(eyes, key=lambda e: e[2] * e[3])
    eye_img = gray[y:y+h, x:x+w]
    return eye_img

def segment_iris(eye_img):
    """
    Approximate Daugman-style segmentation:
    find the pupil (dark circle) and iris (outer boundary) using Hough circles.
    Returns (pupil_circle, iris_circle) as (x, y, radius) or None if not found.
    """
    eye_img = cv2.equalizeHist(eye_img)
    blurred = cv2.GaussianBlur(eye_img, (5, 5), 0)

    # Detect pupil: dark, well-defined circle
    pupil_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=40
    )

    if pupil_circles is None:
        return None, None

    pupil = pupil_circles[0][0]  # (x, y, r)
    px, py, pr = pupil

    # Iris radius is typically ~2-3x the pupil radius
    iris_radius_estimate = int(pr * 2.5)
    iris = (px, py, iris_radius_estimate)

    return pupil, iris

def normalize_iris(eye_img, pupil, iris):
    """
    Daugman's rubber-sheet model: unwrap the annular iris region
    into a fixed-size rectangular image for consistent comparison.
    """
    px, py, pr = pupil
    ix, iy, ir = iris

    height, width = 64, 256  # standard normalized size
    normalized = np.zeros((height, width), dtype=np.uint8)

    for col in range(width):
        theta = 2 * np.pi * col / width
        for row in range(height):
            r_ratio = row / height
            # Interpolate radius between pupil boundary and iris boundary
            radius = pr + r_ratio * (ir - pr)
            x = int(px + radius * np.cos(theta))
            y = int(py + radius * np.sin(theta))

            if 0 <= y < eye_img.shape[0] and 0 <= x < eye_img.shape[1]:
                normalized[row, col] = eye_img[y, x]

    return normalized

def encode_iris(normalized_iris):
    """
    Apply Gabor filters to the normalized iris image to create a binary IrisCode.
    This is the 'fingerprint' we compare between two iris images.
    """
    code_bits = []
    ksize = 9
    sigma = 4.0
    theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 4 orientations
    lambd = 10.0
    gamma = 0.5

    for theta in theta_values:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(normalized_iris, cv2.CV_32F, kernel)
        binary = (filtered > 0).astype(np.uint8)
        code_bits.append(binary)

    iris_code = np.concatenate(code_bits, axis=0)
    return iris_code.flatten()

def hamming_distance(code1, code2):
    """Compare two IrisCodes. 0.0 = identical, 0.5 = random/unrelated."""
    if len(code1) != len(code2):
        return 1.0
    return np.sum(code1 != code2) / len(code1)

def extract_iris_code(frame):
    """Full pipeline: frame -> eye detection -> segmentation -> normalization -> IrisCode."""
    eye_img = detect_eye_region(frame)
    if eye_img is None:
        return None, "No eye detected"

    pupil, iris = segment_iris(eye_img)
    if pupil is None:
        return None, "Could not segment pupil/iris"

    normalized = normalize_iris(eye_img, pupil, iris)
    iris_code = encode_iris(normalized)

    return iris_code, "success"