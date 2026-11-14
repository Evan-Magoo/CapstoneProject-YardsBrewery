"""Shared inspection storage. Compatible with the team's cans/sessions schema."""
import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get('YARDS_METRICS_DB', HERE / 'inspections.db')).resolve()
FAIL_IMAGE_DIR = DB_PATH.parent / 'failed_cans'
SAVE_FAIL_IMAGES = os.environ.get('YARDS_SAVE_FAIL_IMAGES', '1') != '0'

COLUMNS = {
    'session_id': 'TEXT', 'session_start': 'TEXT', 'can_number': 'INTEGER',
    'capture_date': 'TEXT', 'capture_time': 'TEXT', 'verdict': 'TEXT',
    'defect_type': 'TEXT', 'yolo_class': 'TEXT', 'confidence': 'REAL',
    'circularity': 'REAL', 'image_path': 'TEXT', 'processing_ms': 'REAL',
}

def session_id(start):
    return start.strftime('%Y%m%d-%H%M%S-%f')

def connect(path=None):
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=2)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS cans (id INTEGER PRIMARY KEY AUTOINCREMENT, ' +
               ', '.join(f'{name} {kind}' for name, kind in COLUMNS.items()) + ')')
    db.execute('''CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY, start_time TEXT, end_time TEXT, duration TEXT,
        exit_status TEXT, error_message TEXT, total_cans INTEGER, pass_count INTEGER,
        fail_count INTEGER, pass_rate REAL, no_result_count INTEGER)''')
    for table, columns in [('cans', COLUMNS), ('sessions', {'heartbeat': 'REAL', 'state': 'TEXT'})]:
        existing = {row[1] for row in db.execute(f'PRAGMA table_info({table})')}
        for name, kind in columns.items():
            if name not in existing:
                db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {kind}')
    db.commit()
    return db

def heartbeat(start, state='running', error=None, path=None):
    import time
    with closing(connect(path)) as db, db:
        db.execute('''INSERT INTO sessions(session_id,start_time,heartbeat,state,error_message)
            VALUES(?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET
            heartbeat=excluded.heartbeat,state=excluded.state,error_message=excluded.error_message''',
            (session_id(start), start.isoformat(), time.time(), state, error))

def log_can(session_start, can_number, frame_time, verdict, defect_type,
            roi_img, circularity, yolo_class=None, confidence=None, processing_ms=None, path=None):
    image_path = None
    if SAVE_FAIL_IMAGES and verdict == 'FAIL' and roi_img is not None:
        try:
            import cv2
            directory = Path(path or DB_PATH).parent / 'failed_cans' / session_id(session_start)
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f'can-{can_number:06d}.jpg'
            if not cv2.imwrite(str(destination), roi_img):
                raise OSError('Image encoder could not save failed-can image')
            image_path = str(destination)
        except Exception:
            logging.exception('Failed-can image could not be saved; preserving inspection record')
    values = (session_id(session_start), session_start.isoformat(), can_number,
        frame_time.strftime('%Y-%m-%d'), frame_time.strftime('%H:%M:%S.%f'), verdict,
        defect_type, yolo_class, confidence, circularity, image_path, processing_ms)
    with closing(connect(path)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        db.execute(f'''INSERT INTO cans ({','.join(COLUMNS)}) SELECT {','.join('?' for _ in values)}
            WHERE NOT EXISTS (SELECT 1 FROM cans WHERE session_id=? AND can_number=?)''',
            (*values, session_id(session_start), can_number))

def log_session(start, end, total, passed, failed, exit_status, error_message=None, path=None):
    heartbeat(start, 'error' if error_message else 'stopped', error_message, path)
    classified = passed + failed
    with closing(connect(path)) as db, db:
        db.execute('''UPDATE sessions SET end_time=?,duration=?,exit_status=?,error_message=?,
            total_cans=?,pass_count=?,fail_count=?,pass_rate=?,no_result_count=? WHERE session_id=?''',
            (end.isoformat(), str(end-start).split('.')[0], exit_status, error_message,
             total, passed, failed, passed / classified * 100 if classified else 0,
             total-passed-failed, session_id(start)))
