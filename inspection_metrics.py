"""Shared telemetry for the inspection process and read-only dashboard."""
from contextlib import closing
import logging
import os
from pathlib import Path
import sqlite3
import time
import uuid

DB_PATH = Path(os.environ.get('YARDS_METRICS_DB', Path(__file__).resolve().parent / 'data' / 'inspections.sqlite3'))

def connect(path=DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=2)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, started REAL NOT NULL, heartbeat REAL NOT NULL, state TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS inspections (
        session_id TEXT NOT NULL, object_id INTEGER NOT NULL, timestamp REAL NOT NULL,
        outcome TEXT NOT NULL, reason TEXT NOT NULL, predicted_class TEXT,
        confidence REAL, circularity REAL, processing_ms REAL,
        PRIMARY KEY(session_id, object_id));
    ''')
    return db

class MetricsRecorder:
    """Telemetry failures warn rather than interrupt camera/control work."""
    def __init__(self, path=DB_PATH):
        self.path = path
        self.session_id = uuid.uuid4().hex
        self.started = time.time()
        self.last_heartbeat = 0
        self.heartbeat('starting', force=True)

    def _write(self, sql, args):
        try:
            with closing(connect(self.path)) as db, db:
                db.execute(sql, args)
        except (OSError, sqlite3.Error):
            logging.exception('Unable to save inspection metrics')

    def heartbeat(self, state='running', force=False):
        now = time.time()
        if force or now - self.last_heartbeat >= 2:
            self._write('INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)',
                        (self.session_id, self.started, now, state))
            self.last_heartbeat = now

    def record(self, object_id, outcome, reason, predicted_class=None,
               confidence=None, circularity=None, processing_ms=None):
        if outcome not in {'PASS', 'FAIL', 'ERROR'}:
            raise ValueError('outcome must be PASS, FAIL or ERROR')
        self._write('INSERT OR IGNORE INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (self.session_id, object_id, time.time(), outcome, reason,
                     predicted_class, confidence, circularity, processing_ms))
