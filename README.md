# 🪨 CaCO₃ Scale Predictor

<p align="center">
  <img src="assets/preview.png" alt="App preview" width="100%"/>
</p>

<p align="center">
  <a href="https://YOUR-APP.streamlit.app">
    <img src="https://img.shields.io/badge/▶%20Launch%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch App"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge"/>
</p>

> **Hydrodynamic MVP for predicting heterogeneous nucleation (initial crystallisation stage) and CaCO₃ carbonate scale deposition risk in reservoir pore space.**

---

## What it does

An **engineering calculator** that takes formation water chemistry + reservoir properties + well hydrodynamics as inputs, and produces:

| Output | Description |
|--------|-------------|
| **SI(r)** | Saturation Index profile — risk of supersaturation across the drainage zone |
| **R(r)** | Local deposition rate [t/(m³·day)] — where scale accumulates fastest |
| **M(t)** | Total scale mass over the production period [tonnes] |

---

## Physical Model

The model follows four coupled blocks:

### 1 · Water Chemistry

Solubility product corrected for temperature via van't Hoff equation:

$$K_{sp}(T) = 10^{-8.48} \cdot \exp\!\left(\frac{9000}{R}\left(\frac{1}{T_K} - \frac{1}{298.15}\right)\right)$$

CO₃²⁻ concentration from heterogeneous equilibrium:

$$c_{\text{CO}_3^{2-}} = c_{\text{HCO}_3^-} \cdot 10^{\,pH + \log_{10} K_{sp}}$$

Activity coefficients by the extended Debye–Hückel equation:

$$\log_{10} \gamma_i = -\frac{A z_i^2 \sqrt{I}}{1 + B a_i \sqrt{I}}$$

Base Saturation Index:

$$SI_e = \log_{10}\!\frac{a_{\text{Ca}^{2+}} \cdot a_{\text{CO}_3^{2-}}}{K_{sp}}$$

### 2 · Reservoir Properties (FES Block)

Specific pore surface (Kozeny–Carman):

$$S_v = \sqrt{\frac{m^3}{C \cdot k \cdot (1-m)^2 \cdot 10^{-15}}}$$

Heterogeneous nucleation geometric factor:

$$f(\theta) = \frac{(2+\cos\theta)(1-\cos\theta)^2}{4}$$

### 3 · Radial Flow Hydrodynamics

Pressure profile (plane-radial filtration):

$$P(r) = P_e - \frac{P_e - P_w}{\ln(R_e / r_w)} \ln\frac{R_e}{r}$$

Local Saturation Index with pressure correction:

$$SI(r) = SI_e + \lambda (P_e - P(r)), \quad \lambda = 0.1 \; \text{MPa}^{-1}$$

### 4 · Scale Deposition Rate

Heterogeneous nucleation barrier:

$$B_{het}(r) = \frac{\gamma_{cl}^3}{(\ln S(r))^2} \cdot f(\theta)$$

Hydrodynamic suppression factor:

$$F_{flow} = \frac{1}{1 + \beta \cdot v(r)}, \quad \beta = 5 \times 10^4 \; \text{s/m}$$

Deposition rate:

$$R(r) = \frac{S_v}{\sqrt{k \cdot 10^{-15}}} \cdot e^{-\alpha B_{het}(r)} \cdot F_{flow} \cdot \frac{1}{86400 \times 1000}$$

Total mass:

$$M(t) = t \int_{r_w}^{R_e} R(r) \cdot 2\pi r h \; dr$$

---

## Features

- 🌍 **Bilingual UI** — Russian / English toggle
- 🧩 **Step-by-step input wizard** with live validation and range guards
- 📈 **Interactive Plotly charts** — SI(r), P(r), v(r), R(r) with hover tooltips
- 🗺️ **Radial risk heatmap** — colour-coded cross-section of the drainage zone
- 🔬 **Sensitivity analysis** — R vs pH, T, and drawdown (Pₑ − Pw)
- ⏱ **Scale accumulation animation** — M(t) growth over configurable horizon
- 💡 **Expert recommendation engine** — actionable engineering advice based on computed risk

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/caco3-scale-predictor.git
cd caco3-scale-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Repository Structure

```
caco3-scale-predictor/
├── app.py               # Streamlit UI & visualisation
├── model.py             # Core physical model (pure Python + NumPy)
├── requirements.txt
├── README.md
└── assets/
    └── preview.png      # App screenshot for README
```

---

## Applicability & Limitations

| ✅ Applies to | ❌ Not covered |
|---|---|
| CaCO₃ carbonate scale | Other scale types (BaSO₄, CaSO₄, …) |
| Plane-radial filtration | Fracture / dual-porosity media |
| Formation water, T = 20–120 °C | Gas cap / CO₂ degassing effects |
| Sandstone & carbonate reservoirs | Multiphase compositional flow |
| Pressure above bubble point | Below-saturation-pressure conditions |

---

## License

MIT © 2025
