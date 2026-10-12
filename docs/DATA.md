# Data

## 1. Alerts table shipped with the dashboard  (`dashboard/sample_data/stubble_alerts_latest.csv`)
Real model output from the internship project. Computed by `python -m stubble_ews eda-alerts`
(full numbers in `reports/alerts_summary.json`).

| Property | Value |
|---|---|
| Rows (cell-days) | 239,850 |
| Unique H3 cells (resolution 7, ~5 km² each) | 7,995 |
| Parent cells (resolution 5) | 1,692 |
| Dates | 2024-07-02 -> 2024-07-31 (30 days) |
| Latitude / longitude range | 8.25-34.13 N / 68.76-96.34 E (all-India, **not Punjab-only**) |
| Risk bands | Low 234,952 (97.96 %) · Medium 3,867 (1.61 %) · High 1,031 (0.43 %) |
| `p_fire_15d` | min 0.399 · median 0.3995 · max 1.0; 96.3 % of rows < 0.41 |
| Cells ever High | 81 of 7,995 (1.01 %); 22-47 High cells per day (mean 34.4) |

Column dictionary

| Column | Meaning |
|---|---|
| `date` | day the forecast was issued (m/d/yy) |
| `h3_id`, `h3_parent` | H3 cell (res 7) and its res-5 parent |
| `lat`, `lon` | cell centroid |
| `p_fire_15d` | model score: chance of a fire detection in the next 15 days (the Colab export called it `max_p_1_15`; the dashboard renames it) |
| `risk_band` | Low (0.40-0.46), Medium (0.47-0.78), High (>= 0.78) in this file |
| `any_exceeds` | 1 when the score passed the model's operating threshold (identical to High here) |

**Caveat.** July is monsoon season, not the Oct-Nov stubble season, and the cells cover all of India. Use this
file to demo the dashboard, not to draw conclusions about stubble burning in Punjab.

## 2. NASA FIRMS active-fire detections (input of the ML pipeline)
* Source: https://firms.modaps.eosdis.nasa.gov (VIIRS 375 m, MODIS 1 km). Free; needs a registration / MAP_KEY.
* Download Punjab (bbox ~ lat 29.5-32.6, lon 73.8-77.0) as CSV for 2021 -> present and save to `data/raw/`.
* Columns used: `latitude, longitude, acq_date, confidence` (`Lat, Lon, Date, Fire Count` tables also work).
* Quality filter: VIIRS confidence `l` dropped; MODIS confidence < 30 dropped (configurable in `config.py`).
* Per the internship deck: Model 1 used MODIS / NASA FIRMS from 1 Jan 2022 to 30 Jul 2024, daily.

## 3. Sentinel data used by the GEE pipeline (`gee/`)
| Dataset (Earth Engine ID) | Use |
|---|---|
| `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 L2A, 10-20 m) | optical composites; bands B3, B4, B6, B7, B8, B8A, B11, B12; NBR = (B8-B12)/(B8+B12); dNBR = NBR_pre - NBR_post |
| `COPERNICUS/S1_GRD` (Sentinel-1 SAR, 10 m, IW, VV+VH) | rice-flooding signal and burn features |
| `ESA/WorldCover/v200` | cropland mask (class 40) |
| `NOAA/VIIRS/001/VNP14A1`, `MODIS/061/MOD14A1` | fire hotspots (FireMask >= 7) used as training labels |
| `FAO/GAUL/2015/level1,2` | Punjab boundary and districts |

Internship deck (Sentinel-2 results): 5-day windows 21-25 Sep, 1-5 Oct, 11-15 Oct and 1-5 Nov 2024 over Punjab.
