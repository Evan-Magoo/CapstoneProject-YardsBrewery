import cv2
import os
import numpy as np

images = []
count = 0
for filename in os.listdir("test_images"):
    img = cv2.imread(os.path.join("test_images",filename))
    if img is not None:
        images.append(img)

for img in images:
    # Make sure the image was successfully loaded.
    if img is None:
        print("Error: Could not open the image.")
        exit()

    count += 1
    print(f"Image: {count}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=100, minRadius=30, maxRadius=250)

    if circles is not None:
        circles_list = circles[0]

        if circles is not None:
            circle = circles_list[0]
            x = int(circle[0])
            y = int(circle[1])
            radius = int(circle[2])

            print(circle)

            # Draw detected circle
            cv2.circle(img, (x, y), radius, (0, 255, 0), 3)

            # Draw center
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)            
            
            # Add text
            cv2.putText(
                img,
                f"Can top detected - Radius: {radius}px",
                (x - 150, y - radius - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                thickness=4,
                lineType=cv2.LINE_AA
            )

    
            cv2.putText(
                img,
                f"Can top detected - Radius: {radius}px",
                (x - 150, y - radius - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            

    while True:
        cv2.imshow("Can Defect Detection", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()