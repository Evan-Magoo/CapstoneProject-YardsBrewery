import tempfile
import unittest
from pathlib import Path
from inspection_metrics import MetricsRecorder, connect

class MetricsTests(unittest.TestCase):
    def test_duplicate_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.db'
            a = MetricsRecorder(path)
            a.record(1, 'PASS', 'Good can', confidence=.95)
            a.record(1, 'PASS', 'Good can')
            b = MetricsRecorder(path)
            b.record(1, 'FAIL', 'Defective rim')
            b.record(2, 'ERROR', 'No classification returned')
            with connect(path) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM inspections').fetchone()[0], 3)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 2)
                self.assertEqual(db.execute('SELECT confidence FROM inspections WHERE outcome="PASS"').fetchone()[0], .95)

    def test_storage_failure_does_not_stop_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'not-a-directory'
            path.write_text('blocked')
            with self.assertLogs(level='ERROR'):
                recorder = MetricsRecorder(path / 'metrics.db')
                recorder.record(1, 'PASS', 'Good can')

if __name__ == '__main__':
    unittest.main()
