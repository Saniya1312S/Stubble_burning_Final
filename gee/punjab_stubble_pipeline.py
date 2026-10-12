# ==================================================================================
# Punjab Stubble Burning — Full Pipeline (Local-runnable version)
# Auth → Date Clamp → Sentinel-2 → Sentinel-1 → Indices → Crop masks → Hotspots
# → Samples → MV-MLC + K-fold → Probability/Risk → District & Monthly stats
# → 5-km Grid → Clusters → dNBR → Exports
# ==================================================================================

# ---------- Install (run once) ----------
# pip install earthengine-api geemap ee_extra pandas numpy

# ---------- Imports ----------
import os, math, numpy as np, pandas as pd
import ee, geemap
from datetime import date, timedelta

# ---------- Auth ----------
PROJECT_ID = "garbage-detection-468412"
SERVICE_ACCOUNT = "project-17-oct@garbage-detection-468412.iam.gserviceaccount.com"

# Try service account key, fallback to interactive
SA_KEY_PATH = os.environ.get("EE_SA_KEY", "")

try:
    if SA_KEY_PATH and os.path.exists(SA_KEY_PATH):
        credentials = ee.ServiceAccountCredentials(email=SERVICE_ACCOUNT, key_file=SA_KEY_PATH)
        ee.Initialize(credentials, project=PROJECT_ID)
        print("✅ EE initialized with service account:", PROJECT_ID)
    else:
        raise FileNotFoundError("No SA key found, using interactive auth")
except Exception as e:
    print("⚠️ SA auth not used:", e)
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print("✅ EE initialized via interactive auth:", PROJECT_ID)

# ---------- Region: Punjab ----------
def get_punjab():
    gaul1 = ee.FeatureCollection('FAO/GAUL/2015/level1')
    return gaul1.filter(ee.Filter.And(
        ee.Filter.eq('ADM0_NAME','India'),
        ee.Filter.eq('ADM1_NAME','Punjab')
    )).geometry()

REGION = get_punjab()

# ---------- Dates (clamped to today) ----------
def clamp_window(start_d, end_d, fallback_days=30):
    today = date.today()
    end_d = min(end_d, today)
    if start_d >= end_d:
        start_d = end_d - timedelta(days=fallback_days)
    return start_d.isoformat(), end_d.isoformat()

YEAR = date.today().year
PRE_START,  PRE_END  = clamp_window(date(YEAR, 9, 1),  date(YEAR,10,10))
POST_START, POST_END = clamp_window(date(YEAR,10,15),  date(YEAR,12, 5))
OCT_START,  OCT_END  = clamp_window(date(YEAR,10,1),   date(YEAR,10,31), fallback_days=15)
NOV_START,  NOV_END  = clamp_window(date(YEAR,11,1),   date(YEAR,11,30), fallback_days=15)
AUG_START,  SEP_END  = date(YEAR, 8, 1).isoformat(), date(YEAR, 9, 30).isoformat()
FEB_START,  MAR_END  = date(YEAR, 2, 1).isoformat(), date(YEAR, 3, 31).isoformat()
MAY_START,  MAY_END  = date(YEAR, 5, 1).isoformat(), date(YEAR, 5, 31).isoformat()
print("PRE :", PRE_START, "→", PRE_END)
print("POST:", POST_START, "→", POST_END)
print("OCT :", OCT_START, "→", OCT_END)
print("NOV :", NOV_START, "→", NOV_END)

# ---------- FC → pandas ----------
def fc_to_df(fc, props):
    fc = fc.map(lambda f: f.set({
        'longitude': f.geometry().centroid().coordinates().get(0),
        'latitude':  f.geometry().centroid().coordinates().get(1)
    }))
    data = fc.getInfo()
    rows = []
    for feat in data['features']:
        d = {p: feat['properties'].get(p, None) for p in props}
        d['longitude'] = feat['properties'].get('longitude', None)
        d['latitude']  = feat['properties'].get('latitude', None)
        rows.append(d)
    return pd.DataFrame(rows)

# ---------- Sentinel-2 composite (robust fallback) ----------
def s2_comp_robust(start, end, region):
    attempts = [
        dict(col='COPERNICUS/S2_SR_HARMONIZED', cloud=60,  mask=True,  tag='S2_SR strict'),
        dict(col='COPERNICUS/S2_SR_HARMONIZED', cloud=100, mask=True,  tag='S2_SR loose'),
        dict(col='COPERNICUS/S2_HARMONIZED',    cloud=60,  mask=False, tag='S2_TOA strict'),
        dict(col='COPERNICUS/S2_HARMONIZED',    cloud=100, mask=False, tag='S2_TOA loose'),
    ]
    tried = []
    for a in attempts:
        col = (ee.ImageCollection(a['col'])
               .filterDate(start, end)
               .filterBounds(region)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', a['cloud'])))
        n = int((col.size().getInfo() or 0))
        tried.append((a['tag'], n))
        if n > 0:
            if a['mask']:
                def mask_clouds(img):
                    qa = img.select('QA60')
                    return img.updateMask(qa.bitwiseAnd(1<<10).eq(0)).updateMask(qa.bitwiseAnd(1<<11).eq(0))
                col = col.map(mask_clouds)
            img = col.median().clip(region)
            if 'B8' in (img.bandNames().getInfo() or []):
                print(f"✅ S2 composite from {a['tag']} (images: {n})")
                return img
    raise ValueError(f"No S2 imagery {start}..{end}; tried {tried}")

# ---------- Sentinel-1 composite ----------
def s1_comp(start, end, region):
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(region)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode','IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
          .filter(ee.Filter.eq('resolution_meters', 10))).median().clip(region)
    vv = s1.select('VV').rename('S1_VV')
    vh = s1.select('VH').rename('S1_VH')
    return ee.Image.cat([vv, vh, vv.divide(vh.add(1e-6)).rename('S1_VV_VH')])

# ---------- Index helpers ----------
def ndvi(im):
    return im.select('B8').subtract(im.select('B4')).divide(im.select('B8').add(im.select('B4')).add(1e-6))

def nd(a, b):
    return a.subtract(b).divide(a.add(b).add(1e-6))

def nbr(im):
    return nd(im.select('B8'), im.select('B12'))

# ---------- Build composites ----------
print("\n--- Building Sentinel composites ---")
S2_pre  = s2_comp_robust(PRE_START,  PRE_END,  REGION)
S2_post = s2_comp_robust(POST_START, POST_END, REGION)
S2_oct  = s2_comp_robust(OCT_START,  OCT_END,  REGION)
S2_nov  = s2_comp_robust(NOV_START,  NOV_END,  REGION)

USE_S1 = True
S1_augsep = s1_comp(AUG_START, SEP_END, REGION) if USE_S1 else ee.Image([])
S1_post   = s1_comp(POST_START, POST_END, REGION) if USE_S1 else ee.Image([])

# ---------- Spectral indices ----------
NIR, RED, GREEN = S2_post.select('B8'), S2_post.select('B4'), S2_post.select('B3')
SWIR1, SWIR2    = S2_post.select('B11'), S2_post.select('B12')
RE2, RE3, RE4   = S2_post.select('B6'),  S2_post.select('B7'),  S2_post.select('B8A')

NBR_img    = nd(NIR, SWIR2).rename('NBR')
NBRsw  = nd(NIR, SWIR1).rename('NBRswir')
NDSWIR = nd(SWIR1, SWIR2).rename('NDSWIR')
BAIS2  = (SWIR1.multiply(SWIR2)).divide(RE2.add(RE3).add(RE4).multiply(NIR).add(1e-6)).rename('BAIS2')
MIRBI  = SWIR2.multiply(10).subtract(NIR.multiply(9.8)).add(2).rename('MIRBI')
NBRp   = (NBR_img.subtract(nd(SWIR1, GREEN))).rename('NBRplus')

IDX_BANDS = ['NBR','NBRswir','NDSWIR','BAIS2','MIRBI','NBRplus']
IDX_IMG = ee.Image.cat([NBR_img, NBRsw, NDSWIR, BAIS2, MIRBI, NBRp]).clip(REGION)

# ---------- Crop masks ----------
print("\n--- Building crop masks ---")
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
CROPLAND = wc.eq(40).rename('CROPLAND').selfMask().clip(REGION)

NDVI_pre  = ndvi(S2_pre); NDVI_oct = ndvi(S2_oct); NDVI_nov = ndvi(S2_nov); NDVI_post = ndvi(S2_post)

if USE_S1:
    rice_flood = S1_augsep.select('S1_VV_VH').lt(5.5)
else:
    rice_flood = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(20)
rice_ndvi_drop = NDVI_oct.subtract(NDVI_nov).gt(0.1)
RICE_MASK  = CROPLAND.updateMask(rice_flood.And(rice_ndvi_drop)).rename('RICE').selfMask()

S2_febmar = s2_comp_robust(FEB_START, MAR_END, REGION)
S2_may    = s2_comp_robust(MAY_START, MAY_END, REGION)
WHEAT_MASK= CROPLAND.updateMask(
    ndvi(S2_febmar).gt(0.55).And(ndvi(S2_febmar).subtract(ndvi(S2_may)).gt(0.15))
).rename('WHEAT').selfMask()

# ---------- Feature stack ----------
S1_BANDS = ['S1_VV','S1_VH','S1_VV_VH']
FEATURES = IDX_IMG.addBands(S1_post if USE_S1 else ee.Image([])).updateMask(CROPLAND)
ALL_BANDS = IDX_BANDS + (S1_BANDS if USE_S1 else [])
NB = len(ALL_BANDS)

# ---------- Hotspots & Samples ----------
print("\n--- Detecting fire hotspots ---")
def viirs_hotspots(start, end, region):
    viirs = (ee.ImageCollection('NOAA/VIIRS/001/VNP14A1')
             .filterBounds(region).filterDate(start, end))
    if (viirs.size().getInfo() or 0) == 0: return None
    vmask = viirs.max().clip(region).select('FireMask').gte(7).selfMask()
    return vmask.reduceToVectors(geometry=region, scale=1000, geometryType='centroid',
                                 labelProperty='fire', maxPixels=1e9, bestEffort=True)

def modis_hotspots(start, end, region):
    modis = (ee.ImageCollection('MODIS/061/MOD14A1')
             .filterBounds(region).filterDate(start, end))
    if (modis.size().getInfo() or 0) == 0: return None
    mmask = modis.max().clip(region).select('FireMask').gte(7).selfMask()
    return mmask.reduceToVectors(geometry=region, scale=1000, geometryType='centroid',
                                 labelProperty='fire', maxPixels=1e9, bestEffort=True)

def get_fire_hotspots(start, end, region):
    hs, src = viirs_hotspots(start, end, region), 'VIIRS VNP14A1'
    if (hs is None) or ((hs.size().getInfo() or 0) == 0):
        hs, src = modis_hotspots(start, end, region), 'MODIS MOD14A1'
    if (hs is None) or ((hs.size().getInfo() or 0) == 0):
        raise ValueError("No fire hotspots in window.")
    print(f"✅ Hotspots source: {src} (features: {(hs.size().getInfo() or 0)})")
    return hs

hotspots_fc = get_fire_hotspots(POST_START, POST_END, REGION)
burned_zone = hotspots_fc.map(lambda f: f.buffer(120)).union(1)
veg_mask    = NDVI_post.gt(0.4)
safe_zone   = REGION.difference(hotspots_fc.map(lambda f: f.buffer(1000)).union(1).geometry(), 1)

SCALE = 10
N_BURNED, N_UNBURNED = 2500, 2500
burned_pts = FEATURES.sample(region=burned_zone.geometry(), scale=SCALE, numPixels=N_BURNED, seed=13, geometries=True).map(lambda f: f.set('label',1))
unburned_pts = FEATURES.updateMask(veg_mask).sample(region=safe_zone, scale=SCALE, numPixels=N_UNBURNED, seed=37, geometries=True).map(lambda f: f.set('label',0))

# ---------- Samples → pandas ----------
print("\n--- Extracting samples ---")
TRAIN_FRAC, K_FOLDS = 0.7, 5
props = ALL_BANDS + ['label']
df = fc_to_df(burned_pts.merge(unburned_pts).select(props), props).dropna().sample(frac=1.0, random_state=42).reset_index(drop=True)
split = int(len(df)*TRAIN_FRAC)
train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
print(f"Samples: {len(df)} total, {split} train, {len(df)-split} test")

# ---------- MV-MLC + K-fold ----------
def fit_mv_gaussian(X, y):
    Xb, Xu = X[y==1], X[y==0]
    mu_b, mu_u = Xb.mean(axis=0), Xu.mean(axis=0)
    cov_b, cov_u = np.cov(Xb, rowvar=False), np.cov(Xu, rowvar=False)
    reg = 1e-4
    cov_b += np.eye(cov_b.shape[0])*reg; cov_u += np.eye(cov_u.shape[0])*reg
    return dict(mu_b=mu_b, cov_b=cov_b, mu_u=mu_u, cov_u=cov_u,
                prior_b=len(Xb)/len(X), prior_u=len(Xu)/len(X))

def mv_logpdf(X, mu, cov):
    inv = np.linalg.inv(cov); _, logdet = np.linalg.slogdet(cov)
    d = X - mu
    return -0.5*(np.einsum('...i,ij,...j', d, inv, d) + logdet + X.shape[1]*np.log(2*np.pi))

def mv_posteriors(X, p):
    lb = mv_logpdf(X, p['mu_b'], p['cov_b']) + np.log(p['prior_b'])
    lu = mv_logpdf(X, p['mu_u'], p['cov_u']) + np.log(p['prior_u'])
    m  = np.maximum(lb, lu); pb = np.exp(lb-m); pu = np.exp(lu-m); s = pb+pu
    return pb/s, pu/s

def mv_predict(X, p):
    pb, pu = mv_posteriors(X, p); return (pb >= pu).astype(int)

def metrics(y, yhat):
    y, yhat = y.astype(int), yhat.astype(int)
    tp = int(((y==1)&(yhat==1)).sum()); tn = int(((y==0)&(yhat==0)).sum())
    fp = int(((y==0)&(yhat==1)).sum()); fn = int(((y==1)&(yhat==0)).sum())
    acc=(tp+tn)/len(y); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, precision=prec, recall=rec, f1=f1)

print("\n--- Training MV-MLC ---")
Xtr = train_df[ALL_BANDS].astype(float).values; ytr = train_df['label'].astype(int).values
folds = np.array_split(np.arange(len(Xtr)), K_FOLDS); cv=[]
for k in range(K_FOLDS):
    te = folds[k]; tr = np.hstack([folds[i] for i in range(K_FOLDS) if i!=k])
    p = fit_mv_gaussian(Xtr[tr], ytr[tr]); yhat = mv_predict(Xtr[te], p); m = metrics(ytr[te], yhat); m['fold']=k+1; cv.append(m)
cv_df = pd.DataFrame(cv)
print("=== K-fold CV ===")
print(cv_df.to_string())
print(cv_df.agg(['mean','std']).to_string())

p_final = fit_mv_gaussian(Xtr, ytr)
Xte = test_df[ALL_BANDS].astype(float).values; yte = test_df['label'].astype(int).values
print("\n=== Holdout ===")
print(pd.DataFrame([metrics(yte, mv_predict(Xte, p_final))]).to_string())

# ---------- Server-side Probability & Class ----------
print("\n--- Building server-side prediction ---")
mu_b = ee.Array(p_final['mu_b'].tolist()).reshape([NB, 1])
mu_u = ee.Array(p_final['mu_u'].tolist()).reshape([NB, 1])
cov_b_ee = ee.Array(p_final['cov_b'].tolist()); cov_u_ee = ee.Array(p_final['cov_u'].tolist())
inv_b = cov_b_ee.matrixInverse(); inv_u = cov_u_ee.matrixInverse()
log_det_b = ee.Number(np.linalg.slogdet(p_final['cov_b'])[1])
log_det_u = ee.Number(np.linalg.slogdet(p_final['cov_u'])[1])
ln_pb = ee.Number(math.log(p_final['prior_b'])); ln_pu = ee.Number(math.log(p_final['prior_u']))

Ximg_vec = FEATURES.select(ALL_BANDS).toArray().toArray(1)

def qll_vec(Xvec, mu_col, inv, ldet, lnp):
    d  = Xvec.subtract(mu_col)
    dT = ee.Image(d).arrayTranspose()
    q2 = dT.matrixMultiply(inv).matrixMultiply(d)
    q  = q2.arrayGet([0,0])
    return q.multiply(-0.5).subtract(ldet.multiply(0.5)).add(lnp)

ll_b = qll_vec(Ximg_vec, mu_b, inv_b, log_det_b, ln_pb)
ll_u = qll_vec(Ximg_vec, mu_u, inv_u, log_det_u, ln_pu)
mx = ll_b.max(ll_u)
P_BURN = (ll_b.subtract(mx)).exp().divide(
    (ll_b.subtract(mx)).exp().add((ll_u.subtract(mx)).exp()).add(1e-9)
).rename('P_BURN')
MV_BURN = P_BURN.gte(0.5).rename('MV_BURN')

# ---------- Monthly district stats ----------
print("\n--- Computing district stats ---")
gaul2 = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME','Punjab'))

def burned_area(img):
    area = img.multiply(ee.Image.pixelArea())
    def add(f):
        a = area.reduceRegion(ee.Reducer.sum(), f.geometry(), scale=10, maxPixels=1e9, bestEffort=True)
        m2 = ee.Number(a.get(img.bandNames().get(0)))
        return f.set({'burn_area_m2': m2, 'burn_area_ha': m2.divide(10000)})
    return gaul2.map(add)

def month_feats(mimg):
    NIRm, REDm, GREENm = mimg.select('B8'), mimg.select('B4'), mimg.select('B3')
    SW1m, SW2m = mimg.select('B11'), mimg.select('B12')
    RE2m, RE3m, RE4m = mimg.select('B6'), mimg.select('B7'), mimg.select('B8A')
    NBRm   = nd(NIRm, SW2m).rename('NBR')
    NBRswm = nd(NIRm, SW1m).rename('NBRswir')
    NDSWIRm= nd(SW1m, SW2m).rename('NDSWIR')
    BAIS2m = (SW1m.multiply(SW2m)).divide(RE2m.add(RE3m).add(RE4m).multiply(NIRm).add(1e-6)).rename('BAIS2')
    MIRBIm = SW2m.multiply(10).subtract(NIRm.multiply(9.8)).add(2).rename('MIRBI')
    NBRpm  = (NBRm.subtract(nd(SW1m, GREENm))).rename('NBRplus')
    return ee.Image.cat([NBRm,NBRswm,NDSWIRm,BAIS2m,MIRBIm,NBRpm]).addBands(S1_post).updateMask(CROPLAND)

def month_burn(mimg):
    Xv = month_feats(mimg).select(ALL_BANDS).toArray().toArray(1)
    lb = qll_vec(Xv, mu_b, inv_b, log_det_b, ln_pb)
    lu = qll_vec(Xv, mu_u, inv_u, log_det_u, ln_pu)
    mx = lb.max(lu); pb = (lb.subtract(mx)).exp(); pu = (lu.subtract(mx)).exp()
    return pb.divide(pb.add(pu).add(1e-9)).gte(0.5)

OCT_BURN = month_burn(S2_oct).rename('OCT_BURN')
NOV_BURN = month_burn(S2_nov).rename('NOV_BURN')
oct_stats = burned_area(OCT_BURN); nov_stats = burned_area(NOV_BURN)

# ---------- 5-km Grid ----------
CELL_M = 5000
def fishnet_safe(region, cell_m=CELL_M):
    cell_deg = cell_m / 111320.0
    lonlat = ee.Image.pixelLonLat()
    col = lonlat.select('longitude').divide(cell_deg).floor().toInt64()
    row = lonlat.select('latitude').divide(cell_deg).floor().toInt64()
    grid_id = row.multiply(10_000_000).add(col).rename('cell_id')
    return grid_id.reduceToVectors(geometry=region, scale=1000, geometryType='polygon',
                                   labelProperty='cell_id', maxPixels=1e9, bestEffort=True)
GRID = fishnet_safe(REGION, CELL_M)

def grid_stats(grid_fc, prob_img, bin_img):
    area_img = bin_img.multiply(ee.Image.pixelArea())
    def add(f):
        g = f.geometry()
        a = area_img.reduceRegion(ee.Reducer.sum(),  g, scale=10, maxPixels=1e9, bestEffort=True)
        r = prob_img.reduceRegion(ee.Reducer.mean(), g, scale=10, maxPixels=1e9, bestEffort=True)
        return f.set({'burn_area_m2': ee.Number(a.get(bin_img.bandNames().get(0))),
                      'risk_mean':    ee.Number(r.get(prob_img.bandNames().get(0)))})
    return grid_fc.map(add)
GRID_STATS = grid_stats(GRID, P_BURN.updateMask(CROPLAND), MV_BURN)

# ---------- Clusters + dNBR ----------
conn = MV_BURN.selfMask().connectedPixelCount(maxSize=200, eightConnected=True)
CLUSTERS = MV_BURN.updateMask(conn.gte(100))
CLUSTER_POLY = CLUSTERS.reduceToVectors(
    geometry=REGION, scale=10, geometryType='polygon',
    labelProperty='cluster', maxPixels=1e9, bestEffort=True
).map(lambda f: f.simplify(20))
dNBR = nbr(S2_pre).subtract(nbr(S2_post)).rename('dNBR')

# ---------- Exports to Drive ----------
print("\n--- Queuing exports ---")
task_oct = ee.batch.Export.table.toDrive(collection=oct_stats, description='Punjab_Oct_BurnedArea_ByDistrict', fileFormat='CSV'); task_oct.start()
task_nov = ee.batch.Export.table.toDrive(collection=nov_stats, description='Punjab_Nov_BurnedArea_ByDistrict', fileFormat='CSV'); task_nov.start()

area_img = MV_BURN.multiply(ee.Image.pixelArea())
def add_area(f):
    a = area_img.reduceRegion(ee.Reducer.sum(), f.geometry(), scale=10, maxPixels=1e9, bestEffort=True)
    m2 = ee.Number(a.get('MV_BURN'))
    return f.set({'burn_area_m2': m2, 'burn_area_ha': m2.divide(10000)})
district_total = gaul2.map(add_area)
task_dist = ee.batch.Export.table.toDrive(collection=district_total, description='Punjab_BurnedArea_ByDistrict_Total', fileFormat='CSV'); task_dist.start()

task_grid = ee.batch.Export.table.toDrive(collection=GRID_STATS, description='Punjab_Grid5km_Risk_BurnArea', fileFormat='GeoJSON'); task_grid.start()
task_clusters = ee.batch.Export.table.toDrive(collection=CLUSTER_POLY, description='Punjab_Burn_Clusters', fileFormat='GeoJSON'); task_clusters.start()

burn_vectors = MV_BURN.selfMask().reduceToVectors(
    geometry=REGION, scale=10, geometryType='polygon',
    labelProperty='burn', maxPixels=1e9, bestEffort=True
).map(lambda f: f.simplify(20))
task_burnpoly = ee.batch.Export.table.toDrive(collection=burn_vectors, description='Punjab_Predicted_Burn_Polygons', fileFormat='GeoJSON'); task_burnpoly.start()

task_risk = ee.batch.Export.image.toDrive(image=P_BURN.clip(REGION), description='Punjab_Burn_Risk_Prob_10m',
                                          region=REGION, scale=10, maxPixels=1e13, fileFormat='GeoTIFF'); task_risk.start()
task_dnbr = ee.batch.Export.image.toDrive(image=dNBR.clip(REGION), description='Punjab_dNBR_10m',
                                          region=REGION, scale=10, maxPixels=1e13, fileFormat='GeoTIFF'); task_dnbr.start()

print("\n🚀 Exports queued: District (Oct, Nov, Total) CSVs; 5-km Grid GeoJSON; Cluster GeoJSON; Burn Polygons GeoJSON; Risk & dNBR GeoTIFFs.")
print("✅ Pipeline complete.")
