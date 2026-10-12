# app_ee.py — Fire Alerts dashboard (Earth Engine + email)

import os, io, json, ssl, smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
import ee

# ---------------- Page/Secrets ----------------
st.set_page_config(page_title="Fire Alerts (EE)", layout="wide")
def _get_secret(key: str, default: str = "") -> str:
    return os.getenv(key) or st.secrets.get(key, default)

# Thresholds (override via Secrets)
HIGH_THRESH = float(_get_secret("HIGH_THRESH", "0.60"))
MED_THRESH  = float(_get_secret("MED_THRESH", "0.30"))
MIN_PROB    = float(_get_secret("ALERT_MIN_PROB", "0.10"))

# Earth Engine service account auth (strict)
def init_ee_strict():
    svc = st.secrets.get("EE_SERVICE_ACCOUNT")
    key = st.secrets.get("EE_PRIVATE_KEY")
    if not svc or not key:
        st.error("Missing EE_SERVICE_ACCOUNT / EE_PRIVATE_KEY in secrets.")
        st.stop()
    creds = ee.ServiceAccountCredentials(svc, key_data=key)
    ee.Initialize(credentials=creds, project="eighth-opus-452708-f6")

@st.cache_resource
def ee_start():
    init_ee_strict()
    return True

with st.spinner("Initializing Earth Engine…"):
    ee_start()
st.success("✅ EE initialized")

# Assets (change to yours)
PREDICTIONS_ASSET = "projects/eighth-opus-452708-f6/assets/predictions_out2"
FIRMS_ASSET       = "projects/garbage-detection-471720/assets/fire_punjab_with_latlon"

# ---------------- Email (Gmail SMTP) ----------------
SMTP_HOST   = _get_secret("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(_get_secret("SMTP_PORT", "465"))
SMTP_USER   = _get_secret("SMTP_USER", "")  # your gmail/alias
SMTP_PASS   = _get_secret("SMTP_PASS", "")
SENDER_NAME = _get_secret("SENDER_NAME", "Fire Early Warning")
DEFAULT_RECIPS = [r.strip() for r in _get_secret("RECIPIENTS", "").split(",") if r.strip()]

def compose_email(subject: str, body: str, recipients: list, attachments: list[tuple]):
    msg = EmailMessage()
    sender = f"{SENDER_NAME} <{SMTP_USER}>" if SMTP_USER else "alerts@no-reply"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)
    for fname, blob, mime in attachments:
        if not blob: continue
        maintype, subtype = (mime or "application/octet-stream").split("/", 1)
        msg.add_attachment(blob, maintype=maintype, subtype=subtype, filename=fname)
    return msg

def send_email(msg: EmailMessage):
    if not (SMTP_USER and SMTP_PASS):
        raise RuntimeError("Missing SMTP_USER/SMTP_PASS in secrets.")
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

# ---------------- EE helpers ----------------
def ensure_latlon(f):
    lat = ee.Algorithms.If(f.propertyNames().contains('lat'),
                           f.get('lat'), f.geometry().coordinates().get(1))
    lon = ee.Algorithms.If(f.propertyNames().contains('lon'),
                           f.get('lon'), f.geometry().coordinates().get(0))
    return f.set({'lat': ee.Number(lat), 'lon': ee.Number(lon)})

def risk_band(p):
    return (ee.Algorithms.If(ee.Number(p).gte(HIGH_THRESH), "High",
            ee.Algorithms.If(ee.Number(p).gte(MED_THRESH),  "Medium",
            ee.Algorithms.If(ee.Number(p).gte(MIN_PROB),    "Low", "None"))))

@st.cache_data(ttl=600, show_spinner=True)
def load_predictions_df(start_date:str, end_date:str, topn:int=2000) -> pd.DataFrame:
    fc = ee.FeatureCollection(PREDICTIONS_ASSET).map(ensure_latlon)
    # date field auto-detect
    props = fc.first().propertyNames()
    date_col = ee.String(ee.Algorithms.If(props.contains("acq_date"), "acq_date",
                  ee.Algorithms.If(props.contains("date"), "date", "acq_date")))
    # normalize prob column name
    fc = fc.map(lambda f: ee.Algorithms.If(f.propertyNames().contains("p_fire_15d"),
                                           f, ee.Feature(f).set("p_fire_15d", f.get("max_p_1_15")))) \
           .map(lambda f: ee.Feature(f).set("risk_band", risk_band(f.get("p_fire_15d")))) \
           .filter(ee.Filter.date(ee.Date(start_date), ee.Date(end_date).advance(1,"day")))
    # Pull a manageable batch to client
    feats = fc.limit(topn).getInfo()["features"]
    rows = []
    for ft in feats:
        p = ft["properties"]
        lat = p.get("lat"); lon = p.get("lon")
        if lat is None or lon is None: continue
        rows.append({
            "date": p.get(date_col.getInfo(), p.get("acq_date", "")),
            "p_fire_15d": float(p.get("p_fire_15d", np.nan)),
            "risk_band": p.get("risk_band","None"),
            "lat": float(lat), "lon": float(lon),
            "state": p.get("state",""), "district": p.get("district",""),
            "h3_id": p.get("h3_id","")
        })
    return pd.DataFrame(rows)

def color_for(b):
    return {"High":"red","Medium":"orange","Low":"blue"}.get(b,"gray")

def make_map(df: pd.DataFrame) -> folium.Map:
    if df.empty: return folium.Map(location=[30.5,75.5], zoom_start=6)
    m = folium.Map(location=[df["lat"].median(), df["lon"].median()], zoom_start=6, tiles="CartoDB positron")
    for _, r in df.iterrows():
        if pd.isna(r.lat) or pd.isna(r.lon): continue
        popup = folium.Popup(
            f"<b>Risk:</b> {r.risk_band}<br>"
            f"<b>p_fire_15d:</b> {r.p_fire_15d:.3f}<br>"
            f"<b>Date:</b> {r.date}<br>"
            f"<b>State:</b> {r.get('state','-')} | <b>District:</b> {r.get('district','-')}",
            max_width=280
        )
        folium.CircleMarker([r.lat, r.lon], radius=6, color=color_for(r.risk_band),
                            fill=True, fill_opacity=0.9).add_child(popup).add_to(m)
    return m

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")
today = datetime.utcnow().date()
start_date = st.sidebar.date_input("Start date", today - timedelta(days=30))
end_date   = st.sidebar.date_input("End date", today)
sel_bands  = st.sidebar.multiselect("Risk band", ["High","Medium","Low"], default=["High","Medium","Low"])
min_prob   = st.sidebar.slider("Min probability", 0.0, 1.0, MIN_PROB, 0.01)
search_state    = st.sidebar.text_input("State contains", "")
search_district = st.sidebar.text_input("District contains", "")
top_n = st.sidebar.number_input("Max points to load", min_value=100, max_value=5000, value=2000, step=100)

email_ready = bool(SMTP_USER and SMTP_PASS)
st.sidebar.markdown(f"**Email:** {'✅ configured' if email_ready else '⚠️ not configured'}")

# ---------------- Data ----------------
with st.spinner("Loading predictions from Earth Engine…"):
    df = load_predictions_df(str(start_date), str(end_date), topn=int(top_n))

if df.empty:
    st.warning("No predictions in that date range.")
    st.stop()

# band fallback if missing
if "risk_band" not in df.columns:
    def band(p):
        return "High" if p>=HIGH_THRESH else ("Medium" if p>=MED_THRESH else ("Low" if p>=MIN_PROB else "None"))
    df["risk_band"] = df["p_fire_15d"].apply(band)

# apply filters
work = df.copy()
work = work[work["risk_band"].isin(sel_bands)]
work = work[work["p_fire_15d"] >= min_prob]
if search_state.strip() and "state" in work.columns:
    work = work[work["state"].astype(str).str.contains(search_state, case=False, na=False)]
if search_district.strip() and "district" in work.columns:
    work = work[work["district"].astype(str).str.contains(search_district, case=False, na=False)]

# ---------------- Layout/KPIs ----------------
st.title("🔥 Fire Alerts (Earth Engine)")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total alerts", len(work))
c2.metric("High", int((work["risk_band"]=="High").sum()))
c3.metric("Medium", int((work["risk_band"]=="Medium").sum()))
c4.metric("Low", int((work["risk_band"]=="Low").sum()))

# Map
st.subheader("Map")
m = make_map(work)
st_folium(m, height=520)

# Table
st.subheader("Alerts table")
order = [c for c in ["risk_band","p_fire_15d","date","state","district","lat","lon","h3_id"] if c in work.columns]
order += [c for c in work.columns if c not in order]
st.dataframe(work[order].reset_index(drop=True), use_container_width=True)

# ---------------- Downloads + Email ----------------
st.subheader("Download filtered data")
csv_bytes = work.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="alerts_filtered.csv", mime="text/csv")

# GeoJSON
features = []
for _, r in work.iterrows():
    if np.isnan(r.get("lat", np.nan)) or np.isnan(r.get("lon", np.nan)): continue
    props = {k:(None if (isinstance(v,float) and np.isnan(v)) else v) for k,v in r.items()}
    features.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(r["lon"]), float(r["lat"])]},"properties":props})
geojson_bytes = json.dumps({"type":"FeatureCollection","features":features}, ensure_ascii=False).encode("utf-8")
st.download_button("Download GeoJSON", data=geojson_bytes, file_name="alerts_filtered.geojson", mime="application/geo+json")

# Email
st.subheader("📧 Send alert email")
extra = st.text_input("Additional recipients (comma-separated)", "")
recips = DEFAULT_RECIPS + ([r.strip() for r in extra.split(",") if r.strip()] if extra.strip() else [])
recips = sorted(set(recips), key=str.lower)
if not recips:
    st.info("Add recipients in Secrets or above.")

if len(work)==0:
    st.info("No rows in current filter.")
else:
    subject = f"Fire alerts — {datetime.utcnow().date().isoformat()}"
    counts = work["risk_band"].value_counts()
    body = f"""Hi team,

Attached are today's fire alerts from the dashboard.

Counts by band:
{counts.to_string() if not counts.empty else 'N/A'}

Thresholds:
- High: ≥ {HIGH_THRESH:.2f}
- Medium: ≥ {MED_THRESH:.2f}
- Low: ≥ {MIN_PROB:.2f}

Thanks,
{SENDER_NAME}
"""
    atts = [("alerts_filtered.csv", csv_bytes, "text/csv"),
            ("alerts_filtered.geojson", geojson_bytes, "application/geo+json")]
    # Best-effort map HTML
    try:
        html_bytes = m.get_root().render().encode("utf-8")
        atts.append(("alerts_map.html", html_bytes, "text/html"))
    except Exception:
        pass

    disabled = (not email_ready) or (len(recips)==0)
    if st.button("Send email now", disabled=disabled):
        try:
            msg = compose_email(subject, body, recips, atts)
            send_email(msg)
            st.success(f"Email sent to {len(recips)} recipient(s).")
        except Exception as e:
            st.error(f"Email failed: {e}")
