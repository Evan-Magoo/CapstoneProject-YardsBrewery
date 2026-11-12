import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from inspection_metrics import MetricsRecorder

class DashboardTests(unittest.TestCase):
    def test_empty_populated_and_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'db.sqlite3'
            with patch('inspection_metrics.DB_PATH', path):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'dashboard/app.py'), default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertIn('Waiting', app.info[0].value)
                rec = MetricsRecorder(path)
                app.run()
                self.assertFalse(app.exception)
                rec.record(1, 'PASS', 'Good can', confidence=.98, processing_ms=12)
                rec.record(2, 'FAIL', 'Defective rim', circularity=.5, processing_ms=3)
                rec.record(3, 'ERROR', 'No classification returned')
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual([x.value for x in app.metric[:4]], ['3', '1', '1', '1'])
                app.multiselect[0].set_value(['FAIL']).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe[0].value), 1)
