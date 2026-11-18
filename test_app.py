"""
test_app.py
Direct programmatic verification of FastAPI endpoint functions and model pipelines.
"""

import sys
import asyncio
from app import (
    api_get_districts,
    api_predict_district,
    api_predict_national,
    api_get_timeline,
    api_simulate,
    api_export_csv,
    api_get_figure,
    DistrictPredictRequest,
    NationalPredictRequest,
    SimulationRequest
)

async def test_all_endpoints():
    print("="*65, flush=True)
    print("   RUNNING DIRECT ENDPOINT & MODEL PIPELINE TESTS", flush=True)
    print("="*65, flush=True)

    # 1. Test get_districts
    districts = await api_get_districts()
    assert len(districts) == 116, f"Expected 116 districts, got {len(districts)}"
    print(f"[PASS] 1. api_get_districts: {len(districts)} districts loaded across 10 provinces.", flush=True)

    # 2. Test predict_district (Choma)
    req_d = DistrictPredictRequest(district="Choma", season=2027)
    res_d = await api_predict_district(req_d)
    assert res_d["predicted_yield"] > 0
    assert len(res_d["phenological_attention"]) == 12
    print(f"[PASS] 2. api_predict_district (Choma, 2027): Predicted {res_d['predicted_yield']} t/ha, CI: [{res_d['ci_lower']}, {res_d['ci_upper']}], 12 attention windows.", flush=True)

    # 3. Test predict_national
    req_n = NationalPredictRequest(season=2027)
    res_n = await api_predict_national(req_n)
    assert res_n["national_yield_t_ha"] > 0
    assert len(res_n["provinces"]) == 10
    print(f"[PASS] 3. api_predict_national (2027): Weighted national yield = {res_n['national_yield_t_ha']} t/ha, Production = {res_n['total_national_production_mt']:,.0f} MT.", flush=True)

    # 4. Test timeline (CRITICAL REVIEWER REQUIREMENT: 2027 observed yield is strictly None)
    res_t = await api_get_timeline()
    timeline = res_t["timeline"]
    assert len(timeline) == 31
    s2027 = next(t for t in timeline if t["season"] == 2027)
    assert s2027["observed_cfs_yield"] is None, "CRITICAL: 2027 observed yield MUST be None!"
    print(f"[PASS] 4. api_get_timeline: Season 2027 observed yield is strictly None (Forecast Horizon validated).", flush=True)

    # 5. Test simulate
    req_s = SimulationRequest(district="Mazabuka", kdd_override=150.0, gdd_anomaly_pct=-5.0, sar_db_offset=1.5)
    res_s = await api_simulate(req_s)
    assert res_s["predicted_yield"] > 0
    print(f"[PASS] 5. api_simulate (Mazabuka with heat shock KDD=150): Yield = {res_s['predicted_yield']} t/ha, Stress: {res_s['stress_tier']}.", flush=True)

    # 6. Test export CSV
    res_csv = await api_export_csv()
    content = res_csv.body.decode("utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 117 # 1 header + 116 districts
    print(f"[PASS] 6. api_export_csv: Generated CSV with {len(lines)-1} district records.", flush=True)

    # 7. Test get figure
    res_fig = await api_get_figure("figure_thermal_stress_yield_dynamics_corrected.png")
    assert res_fig.status_code == 200
    print(f"[PASS] 7. api_get_figure: Corrected publication figure accessible.", flush=True)

    print("="*65, flush=True)
    print("   ALL TESTS PASSED SUCCESSFULLY (100% PASS RATE)!", flush=True)
    print("="*65, flush=True)

if __name__ == '__main__':
    asyncio.run(test_all_endpoints())
