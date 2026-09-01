import cv2

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Camera couldn't open :(")
    exit()

print("Camera has opened")

while True:
    # ret - boolean indicating frame was correctly read
    # frame - image array
    ret, frame = capture.read()

    if not ret:
        print("Error: Can't receive frame. Exiting!")
        break

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()