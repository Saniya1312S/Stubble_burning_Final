import os
import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
import ee
from datetime import datetime, timedelta

# -----------------------------
# Earth Engine initialization
# -----------------------------
def init_ee():
    """
    Two options:

    A) Service Account (recommended for Streamlit Cloud):
       Set these in Streamlit Secrets:
         EE_SERVICE_ACCOUNT = service-account@project.iam.gserviceaccount.com
         EE_PRIVATE_KEY     = (the entire JSON key content)

    B) OAuth (local dev):
       Comment the service account block and run ee.Authenticate() once locally.
    """
    svc = st.secrets.get("EE_SERVICE_ACCOUNT", None)
    pkey = st.secrets.get("EE_PRIVATE_KEY", None)
    if svc and pkey:
        creds = ee.ServiceAccountCredentials(svc, key_data=pkey)
        ee.Initialize(credentials=creds, project="eighth-opus-452708-f6")
    else:
        # local dev fallback (requires browser auth on first run)
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize(project="eighth-opus-452708-f6")

@st.cache_resource
def ee_start():
    init_ee()
    return True

ee_start()

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Punjab Fire Early Warning", layout="wide")
st.title("🔥 Punjab Fire Early Warning Dashboard")

# EE assets
PREDICTIONS_ASSET = "projects/eighth-opus-452708-f6/assets/predictions_out2"
FIRMS_ASSET       = "projects/garbage-detection-471720/assets/fire_punjab_with_latlon"

@st.cache_data(ttl=3600)
def get_sizes():
    pred_fc = ee.FeatureCollection(PREDICTIONS_ASSET)
    firms_fc = ee.FeatureCollection(FIRMS_ASSET)
    return int(pred_fc.size().getInfo()), int(firms_fc.size().getInfo())

pred_count, firms_count = get_sizes()
st.caption(f"Predictions: **{pred_count}** | FIRMS (Punjab): **{firms_count}**")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    thresh = st.slider("Decision threshold", 0.0, 1.0, 0.40, 0.01)
    horizon_days = st.slider("Horizon (days)", 1, 30, 15)
    buffer_m = st.slider("Match radius (m)", 500, 5000, 2000, 100)
    # date range (string YYYY-MM-DD). We’ll filter predictions by this range.
    today = datetime.utcnow().date()
    start_date = st.date_input("Predictions start date", today - timedelta(days=30))
    end_date   = st.date_input("Predictions end date",   today)
    n_preview  = st.slider("Points on map (preview)", 10, 1000, 200, 10)

# -----------------------------
# EE helpers (server-side)
# -----------------------------
S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
def to_int(b): return ee.Number(ee.Algorithms.If(b, 1, 0))
def to_date_str(d): return ee.Date(d).format("YYYY-MM-dd")

def ensure_latlon(f):
    lat = ee.Algorithms.If(f.propertyNames().contains('lat'),
                           f.get('lat'),
                           f.geometry().coordinates().get(1))
    lon = ee.Algorithms.If(f.propertyNames().contains('lon'),
                           f.get('lon'),
                           f.geometry().coordinates().get(0))
    return f.set({'lat': ee.Number(lat), 'lon': ee.Number(lon)})

def label_predictions(thresh, horizon_days, buffer_m, start_date, end_date):
    pred_fc = ee.FeatureCollection(PREDICTIONS_ASSET).map(ensure_latlon)
    firms_fc = ee.FeatureCollection(FIRMS_ASSET)

    # pick a date field (fallback to 'acq_date')
    first_props = pred_fc.first().propertyNames()
    date_candidates = ee.List(["acq_date","date","pred_date","t0","prediction_date"])
    date_col = ee.String(
        ee.Algorithms.If(
            first_props.contains("acq_date"),
            "acq_date",
            ee.Algorithms.If(
                first_props.contains("date"),
                "date",
                "acq_date"
            )
        )
    )

    # filter prediction date window (string date)
    pred_fc = pred_fc.filter(
        ee.Filter.date(
            ee.Date(str(start_date)), ee.Date(str(end_date + timedelta(days=1)))
        )
    )

    def has_future_fire(pt, t0):
        t1 = ee.Date(t0).advance(horizon_days, "day")
        area = pt.buffer(buffer_m)
        return firms_fc.filterBounds(area).filterDate(t0, t1).size().gt(0)

    def per_feature(f):
        lat = ee.Number(f.get('lat'))
        lon = ee.Number(f.get('lon'))
        pt  = ee.Geometry.Point([lon, lat])
        t0  = ee.Date(f.get(date_col))
        score = ee.Number(f.get("p_fire_15d"))
        pred_bin = score.gte(thresh)
        label = has_future_fire(pt, t0)
        return f.set({
            "date_iso": to_date_str(t0),
            "score": score,
            "pred_bin": pred_bin,
            "label_15d": label
        })

    labeled = pred_fc.map(per_feature)

    # confusion matrix fields
    def add_cf(f):
        y = to_int(f.get("label_15d"))
        p = to_int(f.get("pred_bin"))
        tp = p.multiply(y)
        fp = p.multiply(ee.Number(1).subtract(y))
        fn = (ee.Number(1).subtract(p)).multiply(y)
        tn = (ee.Number(1).subtract(p)).multiply(ee.Number(1).subtract(y))
        return f.set({"TP": tp, "FP": fp, "FN": fn, "TN": tn})

    cf = labeled.map(add_cf)

    # aggregate metrics
    sums = cf.reduceColumns(
        reducer=ee.Reducer.sum().repeat(4),
        selectors=["TP","FP","FN","TN"]
    )
    lst = ee.List(sums.get("sum"))
    TP = ee.Number(lst.get(0)); FP = ee.Number(lst.get(1))
    FN = ee.Number(lst.get(2)); TN = ee.Number(lst.get(3))

    precision = TP.divide(TP.add(FP))
    recall    = TP.divide(TP.add(FN))
    f1        = precision.multiply(recall).multiply(2).divide(precision.add(recall))
    f2        = precision.multiply(recall).multiply(5).divide(ee.Number(4).multiply(precision.add(recall)))
    acc       = TP.add(TN).divide(TP.add(TN).add(FP).add(FN))

    metrics = {
        "TP": TP.getInfo(), "FP": FP.getInfo(), "FN": FN.getInfo(), "TN": TN.getInfo(),
        "precision": precision.getInfo(), "recall": recall.getInfo(),
        "F1": f1.getInfo(), "F2": f2.getInfo(), "accuracy": acc.getInfo()
    }

    # table for alerts (current settings)
    alerts = cf.filter(ee.Filter.eq("pred_bin", True)) \
               .filter(ee.Filter.eq("label_15d", True))
    # pull a preview sample for map/table
    preview = cf.limit(n_preview).getInfo()["features"]

    return metrics, preview, alerts

with st.spinner("Computing metrics & fetching preview…"):
    metrics, preview_feats, alerts_fc = label_predictions(thresh, horizon_days, buffer_m, start_date, end_date)

# -----------------------------
# Metrics display
# -----------------------------
colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Precision", f"{metrics['precision']:.2f}")
colB.metric("Recall",    f"{metrics['recall']:.2f}")
colC.metric("F1",        f"{metrics['F1']:.2f}")
colD.metric("F2",        f"{metrics['F2']:.2f}")
colE.metric("Accuracy",  f"{metrics['accuracy']:.2f}")

st.caption(f"TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}")

# -----------------------------
# Map preview
# -----------------------------
def build_map(preview):
    if not preview:
        m = folium.Map(location=[30.5, 75.5], zoom_start=6, tiles="CartoDB positron")
        return m

    lats = [f["properties"]["lat"] for f in preview if "lat" in f["properties"]]
    lons = [f["properties"]["lon"] for f in preview if "lon" in f["properties"]]
    center = [np.mean(lats), np.mean(lons)] if lats and lons else [30.5, 75.5]

    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    for f in preview:
        p = f["properties"]
        if "lat" not in p or "lon" not in p: continue
        lat, lon = float(p["lat"]), float(p["lon"])
        color = "#d73027" if bool(p.get("pred_bin")) else "#4575b4"
        popup = folium.Popup(
            f"<b>{p.get('date_iso','')}</b><br/>"
            f"score: {float(p.get('score',0)):.2f}<br/>"
            f"pred: {bool(p.get('pred_bin'))} | label: {bool(p.get('label_15d'))}<br/>"
            f"{lat:.4f}, {lon:.4f}",
            max_width=250
        )
        folium.CircleMarker(
            location=[lat, lon], radius=4, color=color, fill=True, fill_opacity=0.8, popup=popup
        ).add_to(m)

    return m

st.subheader("Map preview")
m = build_map(preview_feats)
st_folium(m, width=1100, height=520)

# -----------------------------
# Alerts table
# -----------------------------
st.subheader("Alerts (predicted & matched to FIRMS within horizon)")
def alerts_to_df(fc):
    # Only pull a reasonable number to keep app snappy
    rows = fc.limit(1000).getInfo()["features"]
    recs = []
    for ft in rows:
        p = ft["properties"]
        recs.append({
            "date": p.get("date_iso",""),
            "score": float(p.get("score", 0)),
            "lat": float(p.get("lat", np.nan)),
            "lon": float(p.get("lon", np.nan)),
            "maps": f"https://maps.google.com/?q={float(p.get('lat',0)):.6f},{float(p.get('lon',0)):.6f}"
        })
    return pd.DataFrame(recs)

alerts_df = alerts_to_df(alerts_fc)
st.dataframe(alerts_df, use_container_width=True)
st.download_button("Download alerts (CSV)", data=alerts_df.to_csv(index=False), file_name="alerts.csv", mime="text/csv")

st.caption("Tip: tweak the threshold/horizon/radius in the sidebar to explore different trade-offs.")
