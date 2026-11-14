"""Compatibility recorder: older inspection entry points use the shared team schema."""
import logging
import time
from datetime import datetime
from inspections_db import DB_PATH, connect, heartbeat, log_can, session_id

class MetricsRecorder:
    def __init__(self, path=None):
        self.path = path or DB_PATH
        self.started = datetime.now().astimezone()
        self.session_id = session_id(self.started)
        self.last_heartbeat = 0
        self.heartbeat('starting', force=True)

    def heartbeat(self, state='running', force=False):
        if force or time.time() - self.last_heartbeat >= 2:
            try:
                heartbeat(self.started, state, path=self.path)
                self.last_heartbeat = time.time()
            except Exception:
                logging.exception('Unable to save inspection metrics')

    def record(self, object_id, outcome, reason, predicted_class=None,
               confidence=None, circularity=None, processing_ms=None):
        if outcome not in {'PASS', 'FAIL', 'ERROR'}:
            raise ValueError('outcome must be PASS, FAIL or ERROR')
        try:
            log_can(self.started, object_id, datetime.now().astimezone(), outcome,
                    reason if outcome != 'PASS' else None, None, circularity,
                    predicted_class, confidence * 100 if confidence is not None else None,
                    processing_ms, self.path)
        except Exception:
            logging.exception('Unable to save inspection metrics')
