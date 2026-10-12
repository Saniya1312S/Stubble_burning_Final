# The original repositories compared

All original code is preserved unchanged in this repo. Verified = actually executed during packaging.

| | `Stubble_burning` -> `dashboard/` | `final-burning` -> `earth_engine_apps/alerts_app/` | `Stubble_burning_Final` -> `earth_engine_apps/validation_app/` | `stubble_mainn` -> `gee/` |
|---|---|---|---|---|
| Purpose | CSV alert dashboard + Gmail alerts | Earth Engine alert dashboard + Gmail alerts | Earth Engine dashboard that scores predictions vs FIRMS (precision, recall, F1, F2) | Sentinel-1/2 burned-area classifier |
| Data | `stubble_alerts_latest.csv` (24 MB) | private EE asset `predictions_out2` | same asset + FIRMS asset | public EE datasets |
| Runs for anyone | **Yes (verified)** | No - needs the owner's EE assets | No - needs the owner's EE assets | Needs own EE project |
| Issues found | crashed without `secrets.toml` (fixed by config file); CSV expected in `sample_data/`; needs `pyarrow<18` with its pins | duplicated init logic removed in the other version | duplicate `ee_start` definitions; unused variables; alerts filter on `pred_bin == True` probably never matches (EE returns 1/0) - **untested**; hard-coded project IDs | hard-coded project / service-account IDs |

Fixes applied **without editing any original source line**: `dashboard/.streamlit/secrets.toml`, CSV placed in
`dashboard/sample_data/`, `pyarrow` cap in `dashboard/requirements.txt`.
