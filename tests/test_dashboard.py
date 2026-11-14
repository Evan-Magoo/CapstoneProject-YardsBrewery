import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from inspection_metrics import MetricsRecorder

class DashboardTests(unittest.TestCase):
    def test_empty_populated_filtered_and_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'inspections.db'
            with patch('inspections_db.DB_PATH', path):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'dashboard/app.py'), default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertIn('Waiting', app.info[0].value)
                rec = MetricsRecorder(path)
                rec.heartbeat('running', force=True)
                rec.record(1,'PASS','Good can',confidence=.98,processing_ms=12)
                rec.record(2,'FAIL','Defective rim',circularity=.5,processing_ms=3)
                rec.record(3,'ERROR','No classification returned')
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual([x.value for x in app.metric[:4]], ['3','1','1','50.0%'])
                self.assertTrue(any('Station connected' in x.value for x in app.success))
                app.multiselect[0].set_value(['FAIL']).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe[0].value),1)
                rec.heartbeat('stopped',force=True)
                app.run()
                self.assertTrue(any('Station stopped' in x.value for x in app.info))

                rec.heartbeat('camera_error', force=True)
                app.run()
                self.assertTrue(any('Inspection fault' in x.value for x in app.error))
                from contextlib import closing
                from inspections_db import connect
                with closing(connect(path)) as db, db:
                    db.execute("UPDATE sessions SET state='running', heartbeat=0")
                app.run()
                self.assertTrue(any('No live connection' in x.value for x in app.warning))
