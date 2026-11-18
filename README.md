# Zambia Agro-Satellite Crop Yield Intelligence Platform
## SAR-Augmented Attention CNN-LSTM with Growing Degree Days (GDD) & Killing Degree Days (KDD)

A full-stack web application and scientific modeling platform for maize crop yield forecasting across Zambia at both **district (116 districts)** and **national** scales.

---

### 1. Architecture Overview
The core architecture implements the spatio-temporal deep neural network defined in the study:

1. **Spatio-Temporal Sequence Branch**:
   - **Input**: 12 bi-weekly imagery & climate sequence patches ($32 \times 32$ pixels, 22 multi-modal channels) spanning the November to April growing season.
   - **Channels (22 Bands)**:
     - 12 Sentinel-2 Optical bands ($B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12$)
     - 2 Vegetation Indices ($NDVI, EVI$)
     - 3 Meteorological predictors (CHIRPS precipitation, ERA5 $T_{\max}$, $T_{\min}$)
     - 2 Agrometeorological Thermal Indices:
       - **GDD (Growing Degree Days)**: Cumulative thermal units between $10^\circ\text{C}$ and $30^\circ\text{C}$ ($\sum \max(0, \frac{\min(T_{\max}, 30) + \max(T_{\min}, 10)}{2} - 10)$)
       - **KDD (Killing Degree Days)**: Extreme heat stress accumulation above $30^\circ\text{C}$ ($\sum \max(0, T_{\max} - 30)$)
     - 3 Sentinel-1 SAR bands ($VV$, $VH$, and $VV/VH$ cross-ratio) measuring canopy structural volume, moisture content, and cloud-penetrating biomass.
   - **Time-Distributed 2D CNN Encoder**: Conv2D (32, 64, 128 filters) + SpatialDropout2D (0.40) + GlobalAveragePooling2D.
   - **Bidirectional LSTMs**: 2-layer stacked BiLSTM (96 and 48 units, recurrent dropout = 0.20).
   - **Temporal Attention Mechanism**: Custom score layer $w$ dynamically evaluating phenological vulnerability across the 12 bi-weekly windows, generating context vector $c$ (96 units) and temporal attention weights $\alpha_1, \dots, \alpha_{12}$.

2. **Static Centroid Geographic Prior Branch**:
   - **Input**: District geographic centroid coordinates (Latitude, Longitude).
   - **Dense Mapper**: Dense(16 units, ReLU) + Batch Normalization to account for macro-climate, elevation, and regional soil gradients.

3. **Multimodal Fusion & Regressor Head**:
   - Concatenation of context vector $c$ (96 units) and coordinate embedding (16 units) $\rightarrow$ Dense(32 bottleneck) $\rightarrow$ Dropout(0.45) $\rightarrow$ Maize yield ($t/\text{ha}$ clipped to $[0.5, 4.0]$).

---

### 2. Reviewer Compliance & Figure Corrections
The manuscript figures and application charts have been updated to address reviewer feedback:

1. **Forecast Season (2027)**:
   - **Correction**: In the multi-decade timeline (1997–2027), Season 2027 is strictly treated as an out-of-sample forward-looking forecast season. The `Observed national yield (CFS)` series ends at 2026; for 2027, the observed value is strictly `None` (omitted from plotting).
   - A vertical demarcation (`Forecast Horizon`) demarcates historical seasons from the forecast season.
2. **Neutral Scientific Labels**:
   - Removed biased and evaluative labels such as `Overpredicts under Heat` and `(Corrected)`.
   - Updated legends:
     - `Observed national yield (CFS)` (replaces `Observed Yield (Zambia National)`)
     - `Optical-only baseline` (replaces `Optical Only Baseline (Overpredicts under Heat)`)
     - `SAR + GDD/KDD augmented model` (replaces `GDD+KDD Augmented Model (Corrected)`)
3. **Figure-Embedded Titles Removed**:
   - Canvas-embedded titles (`plt.suptitle` and `ax.set_title` banners) have been removed from the exported figures. Subpanels are cleanly labeled `(a)` and `(b)` with standardized axis descriptions suitable for manuscript captions.
4. **Camera-Ready DPI**:
   - Figures are exported at 300 DPI (`.png`) and vector format (`.pdf`) in `static/figures/`.

---

### 3. Application Features & Interface

- **District Scale Analysis**:
  - Full catalog of all **116 Zambian districts** across all **10 provinces**.
  - Interactive **Leaflet.js** map color-coded by yield tier (green $\ge 2.0$, amber $1.5 - 2.0$, red $< 1.5$ t/ha).
  - KPI metric cards: Predicted yield with 95% Confidence Interval, baseline yield, KDD heat stress, GDD growth units, and SAR $VV/VH$ ratio.
  - Phenological attention chart ($\alpha_t$) showing model weight concentration during reproductive flowering (Windows 7 & 8).
  - Side-by-side benchmark card comparing Optical baseline, Standard CNN-LSTM, and Proposed SAR-Augmented model.
- **National Scale Aggregation**:
  - Production-weighted national yield ($Y_{national} = \frac{\sum A_i Y_i}{\sum A_i}$) and total production (Million Metric Tonnes).
  - National food balance calculation against the national consumption requirement (~2.60 Million MT).
  - Multi-decade historical timeline (1997–2027) with interactive Chart.js.
  - Province summary table (10 provinces) with harvested area, weighted yield, and production volume.
- **Thermal Stress & SAR Scenario Simulator**:
  - Live interactive sliders for KDD heat shock ($0 - 250$ °C-Days), GDD anomaly, Sentinel-1 SAR backscatter offset ($-3$ to $+3$ dB), and CHIRPS precipitation anomaly.
  - Real-time yield re-computation and dynamic attention redistribution.
- **Academic Figures & Export**:
  - Interactive viewer and one-click download buttons for camera-ready PNG & PDF figures.
  - Direct CSV data export for all 116 districts with full agrometeorological attributes.

---

### 4. Running the Application

Double-click `run_app.bat` or run from PowerShell / Command Prompt:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open your browser to:
```
http://127.0.0.1:8000
```

