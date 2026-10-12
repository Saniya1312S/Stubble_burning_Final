# `gee/` – Sentinel-1 / Sentinel-2 burned-area pipeline (Google Earth Engine)

`punjab_stubble_pipeline.py` is a full Earth Engine workflow: Sentinel-2 composites -> burn indices
(NBR, NBRswir, NDSWIR, BAIS2, MIRBI, NBRplus) -> Sentinel-1 VV/VH features -> ESA WorldCover cropland mask ->
VIIRS/MODIS hotspots as training labels -> Multivariate Maximum-Likelihood Classifier with 5-fold CV ->
district / 5-km-grid statistics, clusters, dNBR and exports to Google Drive.



## Before you run it
1. You need a Google Earth Engine account/project. The script hard-codes `PROJECT_ID` and `SERVICE_ACCOUNT`
   near the top – **replace them with your own**, or it will fail for you.
2. `pip install earthengine-api geemap pandas numpy`, then `python punjab_stubble_pipeline.py`
   (falls back to interactive `ee.Authenticate()` when no service-account key is found).
3. Never commit a service-account JSON key (the repo `.gitignore` blocks common names).

Not executed in this repository's CI (needs Earth Engine credentials).
