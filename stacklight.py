 """Control a three-color stack light through three GPIO-driven relay inputs."""

from contextlib import ExitStack
import argparse
import time


class StackLight:
    """Use BCM pin numbers. Set active_high=False for active-low relay inputs."""

    def __init__(self, red_pin, yellow_pin, green_pin, *, active_high=True):
        from gpiozero import OutputDevice

        pins = {"red": red_pin, "yellow": yellow_pin, "green": green_pin}
        if len(set(pins.values())) != len(pins):
            raise ValueError("Each color needs its own GPIO pin.")

        self._outputs = {}
        self._color = "off"
        with ExitStack() as resources:
            for color, pin in pins.items():
                output = OutputDevice(pin, active_high=active_high, initial_value=False)
                resources.callback(output.close)
                resources.callback(output.off)
                self._outputs[color] = output
            self._resources = resources.pop_all()

    def set_color(self, color):
        """Show red, yellow, green, or off. Only one color is active at a time."""
        if color != "off" and color not in self._outputs:
            raise ValueError(f"Unknown stack light color: {color}")
        if color == self._color:
            return
        if self._color != "off":
            self._outputs[self._color].off()
            self._color = "off"
        if color != "off":
            self._outputs[color].on()
            self._color = color

    def show_status(self, status):
        """Map diagnostic PASS to green, FAIL to red, and other states to yellow."""
        if status.startswith("PASS"):
            self.set_color("green")
        elif status.startswith("FAIL"):
            self.set_color("red")
        else:
            self.set_color("yellow")

    def close(self):
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Test each stack light color for two seconds.")
    for color in ("red", "yellow", "green"):
        parser.add_argument(f"--{color}", type=int, required=True, help=f"BCM pin for {color}")
    parser.add_argument("--active-low", action="store_true", help="Relay turns on when its input is low")
    args = parser.parse_args()

    with StackLight(args.red, args.yellow, args.green, active_high=not args.active_low) as light:
        for color in ("red", "yellow", "green"):
            light.set_color(color)
            print(color.upper(), flush=True)
            time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
