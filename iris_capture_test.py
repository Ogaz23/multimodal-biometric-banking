import cv2

cap = cv2.VideoCapture(0)

# Try to set a higher resolution for better iris detail
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Get as close as comfortably possible to the camera, with good even lighting on your face.")
print("Press SPACE to capture, ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Iris Capture Test - Press SPACE to capture", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        break
    elif key % 256 == 32:  # SPACE
        cv2.imwrite("iris_samples/eye_test.jpg", frame)
        print("✅ Saved iris_samples/eye_test.jpg")
        print("Check the image — can you clearly see the iris pattern (not just a dark circle)?")
        break

cap.release()
cv2.destroyAllWindows()