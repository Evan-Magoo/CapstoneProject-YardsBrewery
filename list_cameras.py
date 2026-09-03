import cv2
import time

# Tests first 5 capture devices
def locate_cameras(max_devices=5):
    available_cameras = []
    
    for device_index in range(max_devices):

        camera = cv2.VideoCapture(device_index)
        
        if camera.isOpened():

            success, _ = camera.read()
            if success:
                time.sleep(2)   # Allows cameras time to initialize
                width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Camera found at index {device_index} ({int(width)}x{int(height)})")
                available_cameras.append(device_index)
            else:
                print(f"Device at index {device_index} is present but fails to return frames.")
            
            # Always release the camera resource after testing
            camera.release()
            
    return available_cameras

if __name__ == "__main__":
    print("Scanning for connected cameras...")
    working_devices = locate_cameras()
    print(f"\nScan complete. Active camera indices: {working_devices}")
