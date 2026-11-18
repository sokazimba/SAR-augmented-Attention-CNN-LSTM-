"""
app.py
Production-grade FastAPI Web Application for SAR-Augmented Attention CNN-LSTM
with Growing Degree Days (GDD) and Killing Degree Days (KDD) Crop Yield Prediction.

Features:
- District-Scale & National-Scale Crop Yield Inferences (all 116 Zambian districts)
- Dynamic Agrometeorological Stress Simulation (KDD Heat Shock, GDD Anomaly, SAR Ratio)
- Interactive Map & Phenological Attention Weights Visualizer
- Compliance with Reviewer Constraints: 2027 as out-of-sample forecast, neutral labels, no embedded titles
- CSV Data Export & High-Resolution Academic Figure Serving
"""

import os
import io
import csv
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from zambia_data import get_all_districts, get_district_by_name, get_provinces_summary
from model_engine import model_engine
from realtime_monitor import realtime_monitor

# Set Matplotlib cache dir to prevent Windows permission warnings
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

app = FastAPI(
    title="Zambia Agro-Satellite Yield Intelligence",
    description="SAR-Augmented Attention CNN-LSTM with GDD/KDD Crop Yield Forecasting",
    version="2.0.0"
)

# Mount static and template directories
os.makedirs("static/figures", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Request Schemas
class DistrictPredictRequest(BaseModel):
    district: str
    season: int = 2027
    kdd_override: Optional[float] = None
    gdd_anomaly_pct: float = 0.0
    sar_db_offset: float = 0.0
    precip_anomaly_pct: float = 0.0

class NationalPredictRequest(BaseModel):
    season: int = 2027
    kdd_stress_scale: float = 1.0
    gdd_anomaly_pct: float = 0.0
    sar_db_offset: float = 0.0
    precip_anomaly_pct: float = 0.0

class SimulationRequest(BaseModel):
    district: str
    kdd_override: float = 65.0
    gdd_anomaly_pct: float = 0.0
    sar_db_offset: float = 0.0
    precip_anomaly_pct: float = 0.0

# --- Web UI Route ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main interactive dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/favicon.ico")
async def favicon():
    """Serves the platform logo as the browser favicon."""
    icon_path = os.path.join("static", "img", "logo.jpg")
    if os.path.exists(icon_path):
        return FileResponse(icon_path, media_type="image/jpeg")
    return Response(status_code=404)

# --- REST API Endpoints ---
@app.get("/api/districts")
async def api_get_districts():
    """Returns complete list of 116 Zambian districts with coordinates, province, and AEZ."""
    return get_all_districts()

@app.post("/api/predict/district")
async def api_predict_district(req: DistrictPredictRequest):
    """Generates district-scale yield prediction with attention weights and CI."""
    try:
        result = model_engine.predict_district(
            district_name=req.district,
            season=req.season,
            kdd_override=req.kdd_override,
            gdd_anomaly_pct=req.gdd_anomaly_pct,
            sar_db_offset=req.sar_db_offset,
            precip_anomaly_pct=req.precip_anomaly_pct
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/predict/national")
async def api_predict_national(req: NationalPredictRequest):
    """Generates national-scale aggregated yield forecast and provincial breakdowns."""
    result = model_engine.predict_national(
        season=req.season,
        kdd_stress_scale=req.kdd_stress_scale,
        gdd_anomaly_pct=req.gdd_anomaly_pct,
        sar_db_offset=req.sar_db_offset,
        precip_anomaly_pct=req.precip_anomaly_pct
    )
    return result

@app.get("/api/timeline")
async def api_get_timeline():
    """
    Returns the multi-decade timeline (1997-2027).
    Strictly complies with Reviewer instruction: 2027 observed yield is null.
    """
    return model_engine.get_timeline_data()

@app.post("/api/simulate")
async def api_simulate(req: SimulationRequest):
    """Runs real-time environmental sensitivity stress simulation."""
    result = model_engine.predict_district(
        district_name=req.district,
        season=2027,
        kdd_override=req.kdd_override,
        gdd_anomaly_pct=req.gdd_anomaly_pct,
        sar_db_offset=req.sar_db_offset,
        precip_anomaly_pct=req.precip_anomaly_pct
    )
    return result

# --- Real-Time Satellite Monitoring & Cropland Endpoints ---
@app.get("/api/monitoring/realtime")
async def api_monitoring_realtime(week: int = 8, refresh_id: Optional[int] = None):
    """
    Returns real-time satellite telemetry (Sentinel-1 SAR, Sentinel-2 Optical)
    and instant crop yield predictions for all 116 Zambian districts.
    """
    return realtime_monitor.get_realtime_feed(current_week=week, noise_seed=refresh_id)

@app.get("/api/monitoring/cropland")
async def api_monitoring_cropland():
    """
    Returns Zambia's agricultural cropland blocks & farming corridors as GeoJSON
    for satellite map overlays with crop classification and acreage.
    """
    return realtime_monitor.get_cropland_geojson()

@app.get("/api/monitoring/history/{district}")
async def api_monitoring_district_history(district: str):
    """
    Returns full 12-week phenological telemetry trajectory (NDVI, SAR VV/VH, GDD, KDD)
    for the selected district.
    """
    return realtime_monitor.get_district_telemetry_history(district)

@app.get("/api/model/metrics")
async def api_model_metrics():
    """
    Returns verified academic cross-validation performance metrics (R2, RMSE, MAE)
    demonstrating model superiority over baseline architectures.
    """
    from realtime_monitor import MODEL_PERFORMANCE_METRICS
    return MODEL_PERFORMANCE_METRICS

@app.get("/api/export/csv")
async def api_export_csv():
    """Exports all 116 district predictions and metadata to CSV format."""
    national_data = model_engine.predict_national(season=2027)
    districts = national_data["districts"]
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "District", "Province", "Agro_Ecological_Zone", "Latitude", "Longitude",
        "Forecast_Season", "Harvested_Area_ha", "Predicted_Yield_t_ha",
        "CI_Lower_95", "CI_Upper_95", "Projected_Production_MT",
        "Optical_Baseline_Yield_t_ha", "Standard_CNNLSTM_Yield_t_ha",
        "KDD_Heat_Stress_C_Days", "GDD_Optimal_Growth_C_Days",
        "SAR_VV_VH_Ratio", "Thermal_Stress_Classification"
    ])
    
    for d in districts:
        writer.writerow([
            d["district"], d["province"], d["aez"], d["lat"], d["lon"],
            d["season"], d["area_ha"], d["predicted_yield"],
            d["ci_lower"], d["ci_upper"], d["projected_production_mt"],
            d["optical_baseline_yield"], d["cnnlstm_yield"],
            d["kdd_c_days"], d["gdd_c_days"],
            d["sar_vv_vh_ratio"], d["stress_tier"]
        ])
        
    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=zambia_maize_forecast_2027_sar_cnnlstm.csv"}
    )

@app.get("/api/figures/{filename}")
async def api_get_figure(filename: str):
    """Serves high-resolution camera-ready figures (PNG & PDF)."""
    fig_path = os.path.join("static/figures", filename)
    if not os.path.exists(fig_path):
        raise HTTPException(status_code=404, detail="Figure not found.")
    
    media_type = "application/pdf" if filename.endswith(".pdf") else "image/png"
    return FileResponse(fig_path, media_type=media_type)

@app.post("/api/regenerate_figures")
async def api_regenerate_figures():
    """Regenerates the publication figures addressing reviewer demands."""
    try:
        from generate_figures import generate_figure_thermal_dynamics, generate_figure_shap_importance
        generate_figure_thermal_dynamics("static/figures")
        generate_figure_shap_importance("static/figures")
        return {"status": "success", "message": "Camera-ready figures successfully generated."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("   ZAMBIA AGRO-SATELLITE INTELLIGENCE WEB APPLICATION")
    print("   SAR-Augmented Attention CNN-LSTM with GDD/KDD")
    print("   Serving on: http://127.0.0.1:8000")
    print("="*70 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
