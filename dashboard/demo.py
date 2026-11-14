"""Isolated, clearly labelled sample records for reviewing the UI without hardware."""
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import closing
import shutil
import inspections_db as storage

def demo_database():
    root = Path(__file__).resolve().parents[1]
    path = root / 'data' / 'dashboard-demo' / 'inspections.db'
    if path.exists():
        return path
    start = datetime.now().astimezone() - timedelta(minutes=4)
    storage.heartbeat(start, path=path)
    sample = root / 'images' / 'can_top.jpg'
    archive = path.parent / 'sample-can.jpg'
    if sample.exists():
        shutil.copyfile(sample, archive)
    for number in range(1,49):
        fail = number % 6 == 0
        storage.log_can(start,number,start + timedelta(seconds=number*4),
            'FAIL' if fail else 'PASS', 'DEFECTIVE RIM' if fail else None,
            None, .65 if fail else .96, None if fail else 'good_cans',
            None if fail else 97.2, 42 if fail else 86, path=path)
    if archive.exists():
        with closing(storage.connect(path)) as db, db:
            db.execute('UPDATE cans SET image_path=? WHERE verdict=?',(str(archive),'FAIL'))
    storage.log_session(start,start+timedelta(seconds=192),48,40,8,'Demo data',path=path)
    return path
