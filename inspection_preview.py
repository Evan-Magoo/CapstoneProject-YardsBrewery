"""Publish a throttled preview from the existing camera owner, never open a camera."""
import os
import time
from pathlib import Path

class PreviewPublisher:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.last = 0
        self.previous_latest = None

    def publish(self, frame, name='preview', force=False):
        if not force and time.monotonic() - self.last < .5:
            return
        import cv2
        self.directory.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        preview = cv2.resize(frame, (min(960, width), max(1, int(height * min(960, width) / width))))
        ok, encoded = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise OSError('Could not encode preview')
        destination = self.directory / f'{name}.jpg'
        temporary = destination.with_suffix('.tmp')
        temporary.write_bytes(encoded.tobytes())
        os.replace(temporary, destination)
        if name.startswith('latest-'):
            if self.previous_latest and self.previous_latest != destination:
                self.previous_latest.unlink(missing_ok=True)
            self.previous_latest = destination
        self.last = time.monotonic()
