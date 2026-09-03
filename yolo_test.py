# Use q to exit
import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

capture = cv2.VideoCapture(0)

time.sleep(1)

if not capture.isOpened():
    print("Error: Camera couldn't open :(")
    exit()

print("Camera has opened")

while capture.isOpened():
    # success - boolean indicating frame was correctly read
    # frame - image array
    success, frame = capture.read()

    if not success:
        print("Error: Can't receive frame. Exiting!")
        break

    # object detection
    results = model(frame, verbose=False, stream=True)

    for r in results:
        annoated_frame = r.plot()

    # display frames
    cv2.imshow('YOLO Detection Feed', annoated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()