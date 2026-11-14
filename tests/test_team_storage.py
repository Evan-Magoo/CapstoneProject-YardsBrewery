import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch
from contextlib import closing
import inspections_db as storage

class StorageTests(unittest.TestCase):
    def test_migrate_team_schema_and_live_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'inspections.db'
            with sqlite3.connect(path) as db:
                db.execute('''CREATE TABLE sessions(session_id TEXT PRIMARY KEY,start_time TEXT,end_time TEXT,
                duration TEXT,exit_status TEXT,error_message TEXT,total_cans INTEGER,pass_count INTEGER,
                fail_count INTEGER,pass_rate REAL,no_result_count INTEGER)''')
                db.execute("INSERT INTO sessions(session_id,start_time) VALUES('old','2026-11-13T10:00:00')")
            start = datetime.now().astimezone()
            storage.heartbeat(start,path=path)
            storage.log_can(start,1,start,'FAIL','DEFECTIVE RIM',None,.4,path=path)
            storage.log_can(start,1,start,'FAIL','DEFECTIVE RIM',None,.4,path=path)
            storage.log_session(start,start+timedelta(seconds=10),1,0,1,'Normal Exit (Q)',path=path)
            with closing(storage.connect(path)) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0],2)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM cans').fetchone()[0],1)
                self.assertEqual(db.execute('SELECT state,fail_count FROM sessions WHERE session_id=?',(storage.session_id(start),)).fetchone(),('stopped',1))

    def test_failed_image_and_preview(self):
        import numpy as np
        from inspection_preview import PreviewPublisher
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'inspections.db'
            start = datetime.now().astimezone()
            frame = np.zeros((100,200,3),dtype=np.uint8)
            storage.log_can(start,1,start,'FAIL','BAD CLASS',frame,.9,'bad_cans',95,path=path)
            with closing(storage.connect(path)) as db:
                image = db.execute('SELECT image_path FROM cans').fetchone()[0]
            self.assertTrue(Path(image).is_file())
            publisher = PreviewPublisher(Path(tmp)/'shared_data')
            publisher.publish(frame)
            self.assertTrue((Path(tmp)/'shared_data/preview.jpg').is_file())
            self.assertFalse((Path(tmp)/'shared_data/preview.tmp').exists())
