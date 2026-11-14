# Yards inspection dashboard

Use `can_tester_dbLogging.py` as the camera-owning inspection program and
`dashboard/app.py` as the read-only browser dashboard. Do not run multiple
inspection scripts against the same camera.

## Run on the host (Pi or laptop)

From the repository root, create/activate your Python virtual environment, then:

```sh
python -m pip install -r requirements-dashboard.txt
python -m pip install ultralytics opencv-python numpy
python can_tester_dbLogging.py
```

In another terminal, activate the same environment and run:

```sh
python -m streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501` on the host, or `http://HOST_LAN_IP:8501` on another
phone/tablet/computer on the same reachable network. Campus Wi-Fi isolation or a
host firewall can prevent device-to-device access. Use a permitted shared network;
no public deployment or port forwarding is needed. This dashboard has no login,
so keep it on the intended trusted local network.

The inspection process needs the host's graphical desktop for its OpenCV window.
Camera index is currently 0 and the model is `can_grading_model.pt`. The updated
HEF file in main is not used by this Python path; Hailo inference is separate work.

## Preview without hardware

Use **Explore sample data** in the sidebar. This creates an isolated database in
`data/dashboard-demo/` containing simulated records and a repeated illustrative
can image. It never changes inspection data or opens a camera. Switch it off to
return to the real host database.

## Shared data

Both processes default to the team's `inspections.db` at the repository root.
For a different location, set the same absolute `YARDS_METRICS_DB` in BOTH terminals:

```sh
export YARDS_METRICS_DB=/absolute/path/to/inspections.db
```

The writer upgrades the team's `cans`/`sessions` schema without deleting existing
rows. It adds heartbeat, state and processing time. Session IDs include
microseconds. Confidence uses percentages (0–100). Old sessions remain readable.
Older databases from this branch's former `data/inspections.sqlite3` use a different
schema: retain them as an archive; do not point this dashboard at them.

Failed images are enabled by default and saved under `failed_cans/SESSION/` beside
the database. Set `YARDS_SAVE_FAIL_IMAGES=0` only if image capture is not wanted.
The database stores their file paths. Copy images with the database when moving
archives; paths may need updating on a different host. No automatic archive deletion.

The camera owner publishes a throttled JPEG preview in `shared_data/` beside the
database. Browsers never open a camera or run inference. The latest inspection
thumbnail is keyed to its session and can number, preventing mismatched images.
Heartbeat expires after 10 seconds. Fresh camera previews are hidden when offline.

Laurence's original `streamlit_app.py` remains available and still receives JSON
counts from `can_tester_dbLogging.py`. It has not been redesigned. The older
`can_tester.py` and `can_tester_pi.py` on this branch use a compatibility recorder
for the same database but do not publish camera previews or failed images.

## Meaning of metrics

- One record per session and can number; repeated writes do not increase counts.
- Pass/fail rates use classified PASS+FAIL records, excluding ERROR records.
- Throughput is records divided by session elapsed time, not conveyor speed.
- Confidence is model confidence, not measured accuracy.
- FAIL is a grading result, not confirmation of physical rejection. Valve integration
  remains a placeholder in the inspection script.
- Missing classifications produce ERROR records. Storage failures in the canonical
  inspection script stop that process and are surfaced as a session fault when possible.

## Verify before the demo

1. Run the inspection script and dashboard against the same absolute database.
2. Grade good and failed cans; compare counters and saved records.
3. Check failed image, can number, reason, confidence and capture time.
4. Restart inspection: verify a new session without losing previous records.
5. Stop or disconnect the camera: verify stopped/fault/offline status and no stale preview.
6. Open the host address on a second device and compare results.
7. Confirm a stationary can is not counted repeatedly by the tracker, then test physical
   rejection separately with the controls team.

Local automated checks:

```sh
python -m unittest discover -s tests -v
```

Tests exercise database upgrades, duplicate records, failed image encoding,
preview publication, dashboard filtering, and stopped status. They do not establish
Pi performance, model accuracy, real camera operation, or network reachability.
