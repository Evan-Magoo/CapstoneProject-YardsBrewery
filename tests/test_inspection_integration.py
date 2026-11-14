"""Exercise the real inspection loop using synthetic frames and a stub model.
No camera, GPIO or trained-model inference is used by these tests.
"""
from contextlib import ExitStack, closing
from datetime import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import inspections_db
import can_tester_dbLogging as inspection

class PipelineTests(unittest.TestCase):
    def test_frame_to_database_preview_and_json(self):
        for result_class in ['good_cans', 'bad_cans']:
            with self.subTest(result_class=result_class), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                db_path = root/'inspections.db'
                frame = np.zeros((720,1280,3),dtype=np.uint8)
                cv2.circle(frame,(590,380),150,(255,255,255),-1)
                camera = MagicMock()
                camera.read.side_effect = [(True,frame.copy()),(True,frame.copy())]
                result = SimpleNamespace(probs=SimpleNamespace(top1=0,top1conf=.97),names={0:result_class})
                model = lambda *args, **kwargs: [result]
                stack.enter_context(patch.dict('sys.modules', {'ultralytics':SimpleNamespace(YOLO=lambda path:model)}))
                stack.enter_context(patch.object(inspections_db,'DB_PATH',db_path))
                stack.enter_context(patch.object(inspection,'METRICS_FILE',root/'shared_data/live_metrics.json'))
                stack.enter_context(patch.object(inspection,'get_desktop_path',return_value=root))
                stack.enter_context(patch.object(inspection.time,'sleep'))
                stack.enter_context(patch.object(cv2,'VideoCapture',return_value=camera))
                for method in ['namedWindow','setWindowProperty','imshow','destroyAllWindows']:
                    stack.enter_context(patch.object(cv2,method))
                stack.enter_context(patch.object(cv2,'waitKey',side_effect=[-1,ord('q')]))
                inspection.main()
                with closing(inspections_db.connect(db_path)) as db:
                    rows = db.execute('SELECT verdict,image_path FROM cans').fetchall()
                    state = db.execute('SELECT state FROM sessions').fetchone()[0]
                self.assertEqual(len(rows),1, 'Stationary can should only be counted once')
                self.assertEqual(rows[0][0], 'PASS' if result_class=='good_cans' else 'FAIL')
                if result_class=='bad_cans':
                    self.assertTrue(Path(rows[0][1]).is_file())
                self.assertEqual(state,'stopped')
                self.assertEqual(json.loads((root/'shared_data/live_metrics.json').read_text())['total_inspected'],1)
                self.assertTrue((root/'shared_data/preview.jpg').is_file())
                camera.release.assert_called_once()
