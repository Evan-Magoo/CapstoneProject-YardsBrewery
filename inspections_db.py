"""SQLite log for the can rim inspection station. Two tables:

cans      one row for every good (PASS) and bad (FAIL) can: which session it
          belongs to, when its frame was captured, its number in that session,
          and its defect type.
sessions  one row per run: the same report as the Desktop .txt log (start, end,
          duration, exit status, error, totals, pass rate). Linked to cans by
          session_id.

In the main script:
    import inspection_db
    inspection_db.log_can(session_start, image_count, frame_time, "FAIL", "DEFECTIVE RIM",
                          roi, circularity)
    inspection_db.log_session(session_start, session_end, image_count, pass_count,
                              fail_count, exit_status, error_message)   # in finally

Data goes to inspections.db next to this file. sqlite3 is built into Python.
Run "python inspection_db.py" to print the saved cans and totals per session.
"""
import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- STORAGE: SQLite log, one row per inspected can --------------------------
DB_PATH = HERE / "inspections.db"
SAVE_FAIL_IMAGES = False         # !!!!!!!! True = keep a JPEG (~150 KB) of every failed can
FAIL_IMAGE_DIR = HERE / "failed_cans"
if SAVE_FAIL_IMAGES:
    os.makedirs(FAIL_IMAGE_DIR, exist_ok=True)

COLUMNS = {
    "session_id":    "TEXT",     # e.g. 20261113-215800, one per script start
    "session_start": "TEXT",     # date and time the session started
    "can_number":    "INTEGER",  # 1, 2, 3 ... within this session (image_count)
    "capture_date":  "TEXT",     # date the can's frame was captured (YYYY-MM-DD)
    "capture_time":  "TEXT",     # time the can's frame was captured (HH:MM:SS.mmm)
    "verdict":       "TEXT",     # PASS / FAIL
    "defect_type":   "TEXT",     # DEFECTIVE RIM / BAD CLASS, NULL for a good can
    "yolo_class":    "TEXT",     # NULL when YOLO was skipped
    "confidence":    "REAL",     # percent, NULL when YOLO was skipped
    "circularity":   "REAL",
    "image_path":    "TEXT",     # only for failed cans, when SAVE_FAIL_IMAGES is True
}

db = sqlite3.connect(DB_PATH)
db.execute("PRAGMA journal_mode=WAL")  # safe if the Pi loses power mid-write
db.execute("CREATE TABLE IF NOT EXISTS cans (id INTEGER PRIMARY KEY AUTOINCREMENT, "
           + ", ".join(f"{name} {kind}" for name, kind in COLUMNS.items()) + ")")
db.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id      TEXT PRIMARY KEY,  -- same id as in the cans table
        start_time      TEXT,
        end_time        TEXT,
        duration        TEXT,     -- H:MM:SS
        exit_status     TEXT,     -- Normal Exit (Q) / Interrupted (Ctrl+C) / Crash ...
        error_message   TEXT,     -- NULL when there was no error
        total_cans      INTEGER,  -- image_count
        pass_count      INTEGER,
        fail_count      INTEGER,
        pass_rate       REAL,     -- percent of total_cans, like the .txt report
        no_result_count INTEGER   -- inspected but no final PASS/FAIL
    )
""")
db.commit()

def log_can(session_start, can_number, frame_time, verdict, defect_type,
            roi_img, circularity, yolo_class=None, confidence=None):
    """Save one can.

    session_start  datetime the script started (session_start in the main script)
    can_number     image_count
    frame_time     datetime taken right after cap.read() for this can's frame
    verdict        "PASS" or "FAIL"
    defect_type    "DEFECTIVE RIM", "BAD CLASS", or None for a good can
    """
    image_path = None
    if SAVE_FAIL_IMAGES and verdict == "FAIL":
        import cv2
        image_path = str(FAIL_IMAGE_DIR / f"{frame_time:%Y%m%d-%H%M%S-%f}.jpg")
        cv2.imwrite(image_path, roi_img)
    row = {
        "session_id":    f"{session_start:%Y%m%d-%H%M%S}",
        "session_start": session_start.isoformat(sep=" ", timespec="seconds"),
        "can_number":    can_number,
        "capture_date":  f"{frame_time:%Y-%m-%d}",
        "capture_time":  frame_time.strftime("%H:%M:%S.%f")[:-3],
        "verdict":       verdict,
        "defect_type":   defect_type,
        "yolo_class":    yolo_class,
        "confidence":    None if confidence is None else round(float(confidence), 2),
        "circularity":   round(float(circularity), 4),
        "image_path":    image_path,
    }
    db.execute(f"INSERT INTO cans ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
               list(row.values()))
    db.commit()  # commit every can so nothing is lost on a crash


def log_session(session_start, session_end, image_count, pass_count,
                fail_count, exit_status, error_message=None):
    """Save the end-of-session report (same values as save_session_log)."""
    pass_rate = pass_count / image_count * 100 if image_count else 0.0
    duration = str(session_end - session_start).split(".")[0]  # drop microseconds
    db.execute(
        "INSERT OR REPLACE INTO sessions (session_id, start_time, end_time, duration,"
        " exit_status, error_message, total_cans, pass_count, fail_count, pass_rate,"
        " no_result_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (f"{session_start:%Y%m%d-%H%M%S}",
         session_start.isoformat(sep=" ", timespec="seconds"),
         session_end.isoformat(sep=" ", timespec="seconds"),
         duration, exit_status, error_message, image_count, pass_count,
         fail_count, round(pass_rate, 2), image_count - pass_count - fail_count),
    )
    db.commit()
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    print("Last 5 sessions:")
    for r in db.execute("SELECT session_id, start_time, duration, exit_status, total_cans,"
                        " pass_count, fail_count, pass_rate FROM sessions"
                        " ORDER BY session_id DESC LIMIT 5"):
        print("  ", r)
    print()
    print("Last 10 cans:")
    for r in db.execute("SELECT session_id, can_number, capture_date, capture_time, verdict,"
                        " defect_type, confidence, circularity FROM cans ORDER BY id DESC LIMIT 10"):
        print("  ", r)
    print("\nPer session:")
    for r in db.execute("SELECT session_id, COUNT(*), SUM(verdict='PASS'), SUM(verdict='FAIL'),"
                        " SUM(defect_type='DEFECTIVE RIM'), SUM(defect_type='BAD CLASS')"
                        " FROM cans GROUP BY session_id ORDER BY session_id"):
        print("   {}: {} cans, {} pass, {} fail ({} rim, {} bad class)".format(*r))