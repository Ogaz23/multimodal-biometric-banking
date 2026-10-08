import cv2

# 0 = built-in laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
else:
    print("✅ Webcam opened successfully")
    print("Press 'q' to close the window")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Webcam Test - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()