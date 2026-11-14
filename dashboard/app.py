"""Read-only Streamlit dashboard for the team's shared inspection database."""
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from dashboard.demo import demo_database
from inspections_db import DB_PATH

st.set_page_config(page_title='Yards • Inspection', page_icon='🏭', layout='wide')
st.markdown('''<style>
.stApp {background:#f4f7fb;color:#142c45}
.block-container {max-width:1440px;padding-top:2rem}
h1,h2,h3 {color:#082b50!important;letter-spacing:-.025em}
[data-testid="stMetric"] {background:white;border:1px solid #dee6ef;border-radius:14px;padding:20px 22px;border-top:3px solid #368caf}
[data-testid="stMetricValue"] {color:#082b50}
[data-testid="stTabs"] button {font-weight:600}
.yards-banner {background:#082b50;color:white;padding:28px 32px;border-radius:18px;margin-bottom:24px}
.yards-banner h1 {color:white!important;margin:4px 0;font-size:36px}
.yards-banner p {color:#c0d6eb;margin:0}
.eyebrow {letter-spacing:.18em;font-size:12px;font-weight:700;color:#a9d8ec}
@media(max-width:640px){.block-container{padding:1rem}.yards-banner{padding:20px}.yards-banner h1{font-size:28px}}
</style><div class="yards-banner"><div class="eyebrow">YARDS BREWERY · QUALITY OPERATIONS</div>
<h1>Can inspection overview</h1><p>A clear view of every inspection, from the camera to the record.</p></div>''', unsafe_allow_html=True)
st.sidebar.title('Inspection station')
DEMO = st.sidebar.toggle('Explore sample data', value=False)
if DEMO:
    DB_PATH = demo_database()
    st.warning('DEMO · Simulated results and a repeated illustrative image. No camera or machine is connected to this sample.')
refresh = st.sidebar.toggle('Live updates', value=True)
st.sidebar.caption('Refreshes every 2 seconds. Monitoring only; camera and machine controls stay on the host.')
with st.sidebar.expander('Connect the station'):
    st.caption('On the host, run can_tester_dbLogging.py and this dashboard using the same database path.')

def read_data():
    with closing(sqlite3.connect(DB_PATH.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
        sessions = pd.read_sql_query('SELECT * FROM sessions ORDER BY start_time DESC', db)
        cans = pd.read_sql_query('SELECT * FROM cans ORDER BY id DESC', db)
    return sessions, cans

def show_image(path, caption, unavailable):
    try:
        path = Path(path)
        if path.is_file():
            st.image(path.read_bytes(), caption=caption, width='stretch')
        else:
            st.info(unavailable)
    except (OSError, ValueError):
        st.info(unavailable)

@st.fragment(run_every='2s' if refresh else None)
def dashboard():
    if not DB_PATH.exists():
        st.info('Waiting for inspection data. Start can_tester_dbLogging.py on this host.')
        return
    try:
        sessions, all_cans = read_data()
    except (sqlite3.Error, pd.errors.DatabaseError):
        st.warning('Inspection storage is unavailable or uses an older schema. Start the updated inspection program with the same YARDS_METRICS_DB path.')
        return
    if sessions.empty:
        st.info('No inspection sessions yet. Start the inspection program to begin.')
        return
    selected = st.selectbox('Inspection session', sessions.session_id.tolist(),
        format_func=lambda sid: str(sessions.set_index('session_id').loc[sid, 'start_time']))
    session = sessions.set_index('session_id').loc[selected]
    current = selected == sessions.iloc[0].session_id
    heartbeat = session.get('heartbeat')
    age = time.time() - float(heartbeat) if pd.notna(heartbeat) else float('inf')
    state = session.get('state', 'historical')
    live = current and state == 'running' and age < 10
    if live:
        st.success('● Station connected · waiting for cans or inspecting')
    elif not current:
        st.info('Viewing a previous session · recorded results')
    elif state in ('error', 'camera_error'):
        st.error('Inspection fault · ' + str(session.get('error_message') or 'Check the host inspection program.'))
    elif state == 'stopped':
        st.info('Station stopped · showing saved results')
    elif state == 'starting' and age < 10:
        st.info('Starting camera and model…')
    else:
        st.warning('No live connection · these are saved results, not a running station.')
    df = all_cans[all_cans.session_id == selected].copy()
    passed = int((df.verdict == 'PASS').sum())
    failed = int((df.verdict == 'FAIL').sum())
    errors = len(df) - passed - failed
    for col, label, value in zip(st.columns(4), ['Inspected', 'Passed', 'Failed', 'Pass rate'],
        [len(df), passed, failed, f'{passed/(passed+failed):.1%}' if passed+failed else '—']):
        col.metric(label, value)
    st.caption('PASS / FAIL are inspection decisions. Pass rate is not model accuracy; a failed inspection is not confirmation of physical ejection.')
    live_tab, performance, gallery = st.tabs(['Live inspection', 'Performance & records', 'Failed inspections'])
    with live_tab:
        left, right = st.columns([1.65, 1], gap='large')
        with left:
            st.subheader('Camera preview')
            preview = DB_PATH.parent / 'shared_data' / 'preview.jpg'
            # A previous run must never be shown as a current preview.
            if DEMO:
                show_image(DB_PATH.parent / 'sample-can.jpg', 'Illustrative sample · not a live feed', 'Sample image unavailable.')
            elif live and preview.exists() and time.time() - preview.stat().st_mtime < 10:
                show_image(preview, 'Host camera · refreshed preview', 'Preview unavailable')
            else:
                st.info('Camera preview will appear when the connected inspection program publishes fresh frames.')
        with right:
            st.subheader('Latest inspection')
            if df.empty:
                st.info('No cans inspected in this session yet.')
            else:
                last = df.iloc[0]
                label = str(last.verdict)
                {'PASS': st.success, 'FAIL': st.error}.get(label, st.warning)(f'{label} · Can #{last.can_number}')
                st.write(str(last.defect_type) if pd.notna(last.defect_type) else 'Accepted by inspection')
                st.caption(f'{last.capture_date} · {last.capture_time}')
                a, b = st.columns(2)
                a.metric('Model confidence', f'{last.confidence:.1f}%' if pd.notna(last.confidence) else 'Not evaluated')
                b.metric('Rim circularity', f'{last.circularity:.3f}' if pd.notna(last.circularity) else '—')
                latest = DB_PATH.parent / 'shared_data' / f'latest-{selected}-{last.can_number}.jpg'
                if pd.notna(last.image_path):
                    show_image(last.image_path, 'Illustrative sample' if DEMO else 'Recorded inspection image', 'Image unavailable on this host.')
                elif current and latest.exists() and latest.stat().st_mtime >= pd.Timestamp(session.start_time).timestamp():
                    show_image(latest, 'Latest captured inspection region', 'Image unavailable.')
                else:
                    st.caption('No saved image for this inspection.')
    with performance:
        st.subheader('Session performance')
        start = pd.Timestamp(session.start_time).timestamp()
        end_value = session.get('end_time')
        end = pd.Timestamp(end_value).timestamp() if pd.notna(end_value) else (float(heartbeat) if pd.notna(heartbeat) else start)
        elapsed = max(0, end-start)
        a,b,c = st.columns(3)
        a.metric('Session cans / min', f'{len(df)*60/elapsed:.1f}' if elapsed >= 1 else '—')
        b.metric('Fail rate', f'{failed/(passed+failed):.1%}' if passed+failed else '—')
        c.metric('Unclassified / errors', errors)
        if df.empty:
            st.info('Charts and records appear after the first inspection.')
        else:
            df['captured_at'] = pd.to_datetime(df.capture_date + ' ' + df.capture_time, errors='coerce')
            date_range = st.date_input('Capture dates', value=(df.captured_at.min().date(), df.captured_at.max().date()))
            outcomes = st.multiselect('Show results', ['PASS','FAIL','ERROR'], default=['PASS','FAIL','ERROR'])
            filtered = df[df.verdict.isin(outcomes)]
            if len(date_range) == 2:
                filtered = filtered[(filtered.captured_at.dt.date >= date_range[0]) & (filtered.captured_at.dt.date <= date_range[1])]
            a,b = st.columns(2)
            with a:
                st.markdown('**Inspection activity**')
                if not filtered.empty:
                    counts = filtered.groupby([pd.Grouper(key='captured_at', freq='1min'), 'verdict']).size().unstack(fill_value=0)
                    st.bar_chart(counts, color=[{'PASS':'#24836d','FAIL':'#cb5261','ERROR':'#c89136'}[c] for c in counts.columns])
            with b:
                st.markdown('**Failure reasons**')
                reasons = filtered[filtered.verdict == 'FAIL'].groupby('defect_type').size().rename('Cans')
                if reasons.empty:
                    st.info('No failed inspections in this selection.')
                else:
                    st.bar_chart(reasons, color='#cb5261')
            columns = ['can_number','captured_at','verdict','defect_type','yolo_class','confidence','circularity']
            if 'processing_ms' in filtered:
                columns.append('processing_ms')
            st.dataframe(filtered[columns], hide_index=True, width='stretch')
            st.download_button('Export selected records', filtered[columns].to_csv(index=False), 'yards-inspections.csv', 'text/csv')
            st.caption('Capture dates use the host’s local time. Confidence is a percentage; grading time excludes conveyor travel and ejection.')
    with gallery:
        st.subheader('Failed-can archive')
        failures = df[df.verdict == 'FAIL']
        query = st.text_input('Search can number or defect', placeholder='For example: DEFECTIVE RIM')
        if query:
            failures = failures[failures.can_number.astype(str).str.contains(query, regex=False) | failures.defect_type.fillna('').str.contains(query, case=False, regex=False)]
        if failures.empty:
            st.info('No failed inspections match this session and search.')
        else:
            st.caption(f'{len(failures)} failed inspections · 12 per page')
            page = st.number_input('Archive page', min_value=1, max_value=max(1,(len(failures)+11)//12), step=1)
            for offset in range(0,12,3):
                subset = failures.iloc[(page-1)*12+offset:(page-1)*12+offset+3]
                for col, (_, row) in zip(st.columns(3), subset.iterrows()):
                    with col:
                        with st.container(border=True):
                            st.markdown(f'**Can #{row.can_number} · FAIL**')
                            if pd.notna(row.image_path):
                                show_image(row.image_path, '', 'Image file is unavailable on this host.')
                            else:
                                st.info('No image saved for this inspection.')
                            st.write(str(row.defect_type))
                            st.caption(f'{row.capture_date} · {row.capture_time}')
                            if pd.notna(row.confidence):
                                st.caption(f'Confidence {row.confidence:.1f}%')
    st.divider()
    st.caption('YARDS CAN INSPECTION SYSTEM · Penn State Abington · Team 3')

dashboard()
