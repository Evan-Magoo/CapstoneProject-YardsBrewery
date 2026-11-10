import os
import time

# Define the BCM/Logical GPIO pin number you want to read
# (e.g., GPIO 24)
GPIO_PIN = "24"

SYSFS_PATH = f"/sys/class/gpio/gpio{GPIO_PIN}"

def setup_gpio():
    # 1. Export the pin if it hasn't been exported yet
    if not os.path.exists(SYSFS_PATH):
        try:
            with open("/sys/class/gpio/export", "w") as f:
                f.write(GPIO_PIN)
            # Give Linux a brief moment to create the file structure
            time.sleep(0.1) 
        except PermissionError:
            print("Error: You must run this script with root/sudo privileges to export GPIOs.")
            return False

    # 2. Set the direction of the pin to 'in'
    with open(f"{SYSFS_PATH}/direction", "w") as f:
        f.write("in")
    return True

def read_gpio():
    # 3. Read the actual value (0 or 1)
    with open(f"{SYSFS_PATH}/value", "r") as f:
        value = f.read().strip()
    return value

def cleanup_gpio():
    # 4. Optional: Unexport the pin when done to clean up
    if os.path.exists(SYSFS_PATH):
        with open("/sys/class/gpio/unexport", "w") as f:
            f.write(GPIO_PIN)

if __name__ == "__main__":
    if setup_gpio():
        try:
            print(f"Reading GPIO {GPIO_PIN}. Press Ctrl+C to exit.")
            while True:
                pin_state = read_gpio()
                print(f"Pin {GPIO_PIN} state: {pin_state}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            cleanup_gpio()
