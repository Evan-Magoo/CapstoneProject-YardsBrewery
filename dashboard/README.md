# Rovena's performance dashboard

Branch: `rovena/performance-dashboard`. The dashboard reads saved inspection events;
it does not open a second camera, load YOLO, or operate hardware. Multiple phones,
tablets and computers can view the same host database simultaneously.

## Install on the host (Raspberry Pi or computer)

From your existing repository checkout:

```bash
git fetch origin
git switch rovena/performance-dashboard
python3 -m venv .venv-dashboard
source .venv-dashboard/bin/activate
python -m pip install -r requirements-dashboard.txt
python -m streamlit run dashboard/app.py
```

Run these commands from the repository root so `.streamlit/config.toml` is loaded.
On Windows activate with `.venv-dashboard\Scripts\activate` instead.
The dashboard environment only needs Streamlit and pandas, not the camera packages.

In a **separate terminal**, use your existing inspection environment and launch
**one** of the instrumented scripts, for example:

```bash
python3 can_tester_pi.py
```

Keep the normal camera/OpenCV desktop session available. The dashboard doesn't
make the inspection script headless. `can_tester.py` is instrumented too.
Your existing YOLO, OpenCV and NumPy installation is still required for inspection.

## Open from another device

1. Connect the host and viewing devices to the same reachable local network.
2. On the Pi, run `hostname -I` and use its LAN address (for example `192.168.1.42`).
3. On a phone, tablet or laptop open **http://192.168.1.42:8501**, substituting
   the real host address. On the host itself use **http://localhost:8501**.

`0.0.0.0` is the listening address, not the address to type into a browser.
The host must remain on and both programs must remain running. Ctrl+C stops the
web dashboard. No cloud deployment or internet connection is needed after install.
The config keeps Streamlit's default CORS and XSRF protections enabled.

If another device cannot connect, check the IP and port, allow inbound TCP 8501
on the host firewall for your trusted LAN, and check with campus IT for Wi-Fi
client isolation or VLAN restrictions. Sharing the same Wi-Fi name alone may not
allow device-to-device connections. Do not port-forward this unauthenticated
read-only dashboard to the public internet. Anyone who can reach it can view and
export inspection records.

## Metrics and storage

- Events are saved in `data/inspections.sqlite3` (ignored by Git).
- Each inspection-program start creates a new session. Choose sessions in the UI.
- Each graded tracker ID is saved once per session, including unclassified errors.
- Pass rate = PASS / (PASS + FAIL). Errors are shown separately.
- Throughput = recorded inspections / session minutes up to the last heartbeat;
  this includes idle time. It is not a calibrated conveyor speed.
- Mean grading time covers the grading block, not camera capture or air rejection.
- Confidence is the classifier score, not measured accuracy. Rim-only failures
  have no classifier score. FAIL does not confirm physical removal from the belt.
- The graph bins inspection events into UTC minutes. CSV exports follow the
  selected session and result filters.
- A heartbeat older than 10 seconds is shown as disconnected/stale; a slow or
  stalled model can also delay it. Closing the camera normally reports stopped;
  camera-read failures report camera_error. Crashes eventually become stale.
- Refresh happens every 3 seconds while the browser session is active.

The existing tracker may assign new IDs to a can that leaves and re-enters view;
this dashboard preserves its grading events rather than asserting unique physical
can counts. It does not change classification thresholds or rejection behavior.

For a custom DB location, set `YARDS_METRICS_DB` to the same absolute path in
**both** terminals before starting the scripts. Keep the SQLite file on the host's
local disk, not a network share. Telemetry write failures are printed in the
inspection terminal rather than stopping inspection. Back up the database when
both programs are stopped. No automatic deletion/retention is configured.

## Checks

```bash
python -m unittest discover -s tests -p 'test_metrics.py'
```

Hardware-free UI testing can use Streamlit's AppTest. Bench validation with a real
camera and a second LAN device is still required before the class demonstration.
