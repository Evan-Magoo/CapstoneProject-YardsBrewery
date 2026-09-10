import cv2
import numpy as np

KNOWN_OD_MM = 56.9
pixels_per_mm = None

# Index 0 for the laptop webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection helps isolate both the inner and outer rim lines
    edges = cv2.Canny(blurred, 40, 120)
    
    # Close small gaps in the edge lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # RETR_LIST retrieves all contours without parent/child hierarchy
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    display = frame.copy()

    # Filter out noise and sort by largest area
    valid_contours = [c for c in contours if cv2.contourArea(c) > 3000 and len(c) >= 5]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    # We need at least 2 distinct contours (outer lip and inner plug)
    if len(valid_contours) >= 2:
        outer_contour = valid_contours[0]
        inner_contour = valid_contours[1]

        (cx_out, cy_out), (d1_out, d2_out), angle_out = cv2.fitEllipse(outer_contour)
        (cx_in, cy_in), (d1_in, d2_in), angle_in = cv2.fitEllipse(inner_contour)

        # Average the axes to get the pixel diameter
        outer_px_dia = (d1_out + d2_out) / 2.0
        inner_px_dia = (d1_in + d2_in) / 2.0

        # Draw the detected bounds
        cv2.ellipse(display, ((cx_out, cy_out), (d1_out, d2_out), angle_out), (255, 0, 0), 2)
        cv2.ellipse(display, ((cx_in, cy_in), (d1_in, d2_in), angle_in), (0, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            pixels_per_mm = outer_px_dia / KNOWN_OD_MM
            print(f"Calibrated: {pixels_per_mm:.2f} pixels per mm")

        if pixels_per_mm:
            # Calculate physical measurements in millimeters
            outer_mm = outer_px_dia / pixels_per_mm
            plug_mm = inner_px_dia / pixels_per_mm
            flange_mm = (outer_mm - plug_mm) / 2.0

            # Tolerances from the Dimension Key table
            plug_pass = 52.15 <= plug_mm <= 52.65     # 52.40 ± 0.25 mm
            flange_pass = 1.85 <= flange_mm <= 2.35   # 2.10 ± 0.25 mm
            
            is_good = plug_pass and flange_pass
            
            status_text = "PASS" if is_good else "FAIL"
            status_color = (0, 255, 0) if is_good else (0, 0, 255)
            cv2.putText(display, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            plug_color = (0, 255, 0) if plug_pass else (0, 0, 255)
            cv2.putText(display, f"Plug (A): {plug_mm:.2f} mm", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, plug_color, 2)
            
            flange_color = (0, 255, 0) if flange_pass else (0, 0, 255)
            cv2.putText(display, f"Flange (B): {flange_mm:.2f} mm", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, flange_color, 2)
        else:
            cv2.putText(display, "Press 'C' to calibrate OD (56.9mm)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Dimensional Inspection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()