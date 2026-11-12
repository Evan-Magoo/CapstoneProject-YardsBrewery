"""LAN dashboard: no camera, model or GPIO imports in this process."""
from contextlib import closing
from pathlib import Path
import sys
import time
import sqlite3
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from inspection_metrics import DB_PATH

st.set_page_config(page_title='Yards | Performance', page_icon='📊', layout='wide')
st.title('Yards Brewery')
st.caption('CAN INSPECTION • PERFORMANCE DASHBOARD')
st.sidebar.header('Viewing options')
refresh = st.sidebar.toggle('Auto-refresh every 3 seconds', value=True)
st.sidebar.caption('Read-only monitoring. Keep the inspection program running on the host.')

@st.fragment(run_every='3s' if refresh else None)
def dashboard():
    if not DB_PATH.exists():
        st.info('Waiting for inspection data. Start can_tester.py or can_tester_pi.py on the host.')
        return
    try:
        with closing(sqlite3.connect(DB_PATH.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
            sessions = pd.read_sql_query('SELECT * FROM sessions ORDER BY started DESC', db)
            if sessions.empty:
                st.info('No inspection sessions yet.')
                return
            session = st.selectbox('Inspection session (UTC)', sessions.id.tolist(),
                format_func=lambda sid: pd.to_datetime(sessions.set_index('id').loc[sid, 'started'], unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S') + ' · ' + sid[:8])
            df = pd.read_sql_query('SELECT * FROM inspections WHERE session_id=? ORDER BY timestamp', db, params=(session,))
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        st.warning(f'Cannot read metrics yet: {exc}')
        return
    row = sessions.set_index('id').loc[session]
    age = max(0, time.time() - row.heartbeat)
    if row.state == 'running' and age < 10:
        st.success('Inspection process connected')
    else:
        st.warning(f'No live heartbeat · last state: {row.state} · last update {age:.0f}s ago')
    st.caption('Session timestamps are UTC. PASS/FAIL are model decisions, not verified accuracy or confirmed physical ejections.')
    total = len(df)
    passed = int((df.outcome == 'PASS').sum())
    failed = int((df.outcome == 'FAIL').sum())
    errors = int((df.outcome == 'ERROR').sum())
    cols = st.columns(4)
    for col, label, value in zip(cols, ['Inspections', 'Passed', 'Failed', 'Unclassified / errors'], [total, passed, failed, errors]):
        col.metric(label, value)
    classified = passed + failed
    elapsed = max(0, row.heartbeat - row.started)
    cols = st.columns(3)
    cols[0].metric('Pass rate (classified)', f'{passed / classified:.1%}' if classified else '—')
    cols[1].metric('Session inspections / min', f'{total * 60 / elapsed:.1f}' if elapsed >= 1 else '—')
    ms = df.processing_ms.dropna()
    cols[2].metric('Mean grading time', f'{ms.mean():.0f} ms' if len(ms) else '—')
    if df.empty:
        st.info('Connected session has no recorded inspections yet.')
        return
    df['time_utc'] = pd.to_datetime(df.timestamp, unit='s', utc=True)
    left, right = st.columns(2)
    with left:
        st.subheader('Inspection activity')
        counts = df.groupby([pd.Grouper(key='time_utc', freq='1min'), 'outcome']).size().unstack(fill_value=0)
        st.bar_chart(counts, color=[{'PASS': '#237a57', 'FAIL': '#bb4239', 'ERROR': '#b58b28'}[c] for c in counts.columns])
        st.caption('Recorded inspection events per UTC minute.')
    with right:
        st.subheader('Failure reasons')
        reasons = df[df.outcome == 'FAIL'].groupby('reason').size().rename('Count')
        if reasons.empty:
            st.info('No failed inspections in this session.')
        else:
            st.bar_chart(reasons, color='#bb4239')
    st.subheader('Inspection log')
    outcomes = st.multiselect('Show results', ['PASS', 'FAIL', 'ERROR'], default=['PASS', 'FAIL', 'ERROR'])
    filtered = df[df.outcome.isin(outcomes)].drop(columns=['timestamp', 'session_id'])
    st.dataframe(filtered.sort_values('time_utc', ascending=False), hide_index=True, use_container_width=True)
    st.download_button('Download filtered CSV', filtered.to_csv(index=False), 'yards-inspections.csv', 'text/csv')
    st.caption('Confidence is recorded as a fraction (0–1); grading time excludes conveyor travel and valve actuation.')

dashboard()
