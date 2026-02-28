"""
CaCO₃ Scale Deposition Predictor  —  app.py  (v2.1)
Streamlit-приложение. Требует model.py (v2.1) в той же директории.
"""

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import run_model

# ═══════════════════════════════════════════════════════════════════════
# ЛОКАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════
LANG = {
    "ru": {
        "subtitle":   "Прогнозирование риска карбонатного солеотложения в поровом пространстве коллектора",
        "lang_btn":   "🇬🇧 English",
        # Шаги
        "step1": "Химия воды", "step2": "ФЕС коллектора", "step3": "Гидродинамика",
        "next": "Далее →", "back": "← Назад", "calc": "▶  Рассчитать",
        "edit": "← Изменить параметры",
        # Ввод — химия
        "T_lbl":    "Температура пласта (°C)",
        "T_hlp":    "Диапазон: 20–120 °C",
        "pH_lbl":   "pH пластовой воды",
        "pH_hlp":   "Диапазон: 1–14",
        "CCa_lbl":  "Концентрация Ca²⁺ (мг/л)",
        "CHCO3_lbl":"Концентрация HCO₃⁻ (мг/л)",
        "CNaK_lbl": "Концентрация Na⁺ + K⁺ (мг/л)",
        "CCl_lbl":  "Концентрация Cl⁻ (мг/л)",
        # Ввод — ФЕС
        "m_lbl":    "Пористость (доли)",
        "m_hlp":    "Диапазон: 0.01–0.99",
        "rock_lbl": "Тип коллектора",
        "rock_opt": ["Песчаник", "Карбонат"],
        "rock_val": ["sandstone", "carbonate"],
        "k_lbl":    "Проницаемость по воде (мД)",
        # Ввод — гидродинамика
        "Qw_lbl":   "Дебит воды Q_w (м³/сут)",
        "Re_lbl":   "Радиус контура питания R_e (м)",
        "rw_lbl":   "Радиус скважины r_w (м)",
        "h_lbl":    "Эффективная толщина пласта h (м)",
        "Pe_lbl":   "Пластовое давление P_e (МПа)",
        "Pw_lbl":   "Забойное давление P_w (МПа)",
        "t_lbl":    "Период эксплуатации t (сут)",
        "rpbz_lbl": "Радиус призабойной зоны r_ПЗП (м)",
        "depr_lbl": "Депрессия ΔP",
        # Ошибки валидации
        "err_T":   "⛔ Температура должна быть 20–120 °C",
        "err_pH":  "⛔ pH должен быть 1–14",
        "err_m":   "⛔ Пористость должна быть в диапазоне 0–1",
        "err_Pw":  "⛔ Забойное давление должно быть меньше пластового",
        "err_rw":  "⛔ Радиус скважины должен быть меньше R_e",
        # KPI
        "kpi_SI_lbl":   "Макс. индекс насыщения SI",
        "kpi_R_lbl":    "Макс. скорость осаждения",
        "kpi_M_lbl":    "Масса осадка в ПЗП за период",
        "kpi_R_unit":   "кг/(м³·сут)",
        "kpi_M_unit":   "кг",
        "si_neg":  "Недонасыщено",
        "si_low":  "Слабое пересыщение",
        "si_mid":  "Умеренный риск",
        "si_high": "Высокий риск!",
        # Графики
        "tab_SI":  "SI(r)",
        "tab_P":   "P(r)",
        "tab_v":   "v(r)",
        "tab_R":   "R(r)",
        "r_ax":    "Радиус r (м)",
        "SI_ax":   "SI(r)",
        "P_ax":    "P (МПа)",
        "v_ax":    "v (м/с)",
        "R_ax":    "R (кг/(м³·сут))",
        "heatmap_title": "Тепловая карта зоны риска",
        "heatmap_cap":   "Радиальная схема пласта. Центр — скважина. Цвет по R(r).",
        "wellbore": "Скважина",
        # Чувствительность
        "sens_title": "Анализ чувствительности",
        "sens_note":  "Изменение R(r_w) при вариации параметра. Остальное — как введено.",
        "sens_btn":   "Запустить анализ",
        "sens_run":   "Анализ чувствительности...",
        "sens_pH":    "R vs pH",
        "sens_T":     "R vs T (°C)",
        "sens_Pw":    "R vs P_w (МПа)",
        # Анимация
        "anim_title": "Накопление осадка M_ПЗП(t)",
        "anim_years": "Горизонт моделирования (лет)",
        "anim_btn":   "Показать M(t)",
        "anim_yax":   "M_ПЗП (кг)",
        "anim_xax":   "Время (лет)",
        # Рекомендации
        "rec_title": "Инженерные рекомендации",
        "rec_none":  "✅ Риск солеотложения минимален. Мониторинг рекомендован в штатном режиме.",
        # Детали
        "det_title": "Детали расчёта",
        "det_SI_e":  "SI_e (пластовые условия)",
        "det_Sv":    "Удельная поверхность S_v (м²/м³)",
        "det_M_all": "Масса осадка — весь пласт (кг)",
        "det_M_pbz": "Масса осадка — ПЗП (кг)",
    },
    "en": {
        "subtitle":   "Carbonate scale deposition risk prediction in reservoir pore space",
        "lang_btn":   "🇷🇺 Русский",
        "step1": "Water Chemistry", "step2": "Reservoir Properties", "step3": "Hydrodynamics",
        "next": "Next →", "back": "← Back", "calc": "▶  Calculate",
        "edit": "← Edit parameters",
        "T_lbl":    "Reservoir Temperature (°C)",
        "T_hlp":    "Range: 20–120 °C",
        "pH_lbl":   "Formation Water pH",
        "pH_hlp":   "Range: 1–14",
        "CCa_lbl":  "Ca²⁺ Concentration (mg/L)",
        "CHCO3_lbl":"HCO₃⁻ Concentration (mg/L)",
        "CNaK_lbl": "Na⁺ + K⁺ Concentration (mg/L)",
        "CCl_lbl":  "Cl⁻ Concentration (mg/L)",
        "m_lbl":    "Porosity (fraction)",
        "m_hlp":    "Range: 0.01–0.99",
        "rock_lbl": "Reservoir Type",
        "rock_opt": ["Sandstone", "Carbonate"],
        "rock_val": ["sandstone", "carbonate"],
        "k_lbl":    "Water Permeability (mD)",
        "Qw_lbl":   "Water Flow Rate Q_w (m³/day)",
        "Re_lbl":   "Drainage Radius R_e (m)",
        "rw_lbl":   "Wellbore Radius r_w (m)",
        "h_lbl":    "Net Pay Thickness h (m)",
        "Pe_lbl":   "Reservoir Pressure P_e (MPa)",
        "Pw_lbl":   "Bottom-hole Pressure P_w (MPa)",
        "t_lbl":    "Production Period t (days)",
        "rpbz_lbl": "Near-wellbore Zone Radius r_NWZ (m)",
        "depr_lbl": "Drawdown ΔP",
        "err_T":   "⛔ Temperature must be 20–120 °C",
        "err_pH":  "⛔ pH must be 1–14",
        "err_m":   "⛔ Porosity must be between 0 and 1",
        "err_Pw":  "⛔ Bottom-hole pressure must be less than reservoir pressure",
        "err_rw":  "⛔ Wellbore radius must be less than R_e",
        "kpi_SI_lbl":   "Max Saturation Index SI",
        "kpi_R_lbl":    "Max Deposition Rate",
        "kpi_M_lbl":    "Scale Mass in NWZ over Period",
        "kpi_R_unit":   "kg/(m³·day)",
        "kpi_M_unit":   "kg",
        "si_neg":  "Undersaturated",
        "si_low":  "Slight supersaturation",
        "si_mid":  "Moderate risk",
        "si_high": "High risk!",
        "tab_SI":  "SI(r)",
        "tab_P":   "P(r)",
        "tab_v":   "v(r)",
        "tab_R":   "R(r)",
        "r_ax":    "Radius r (m)",
        "SI_ax":   "SI(r)",
        "P_ax":    "P (MPa)",
        "v_ax":    "v (m/s)",
        "R_ax":    "R (kg/(m³·day))",
        "heatmap_title": "Risk Zone Heatmap",
        "heatmap_cap":   "Radial reservoir cross-section. Center — wellbore. Color by R(r).",
        "wellbore": "Wellbore",
        "sens_title": "Sensitivity Analysis",
        "sens_note":  "Change in R(r_w) when varying one parameter. Others held constant.",
        "sens_btn":   "Run Analysis",
        "sens_run":   "Running sensitivity analysis...",
        "sens_pH":    "R vs pH",
        "sens_T":     "R vs T (°C)",
        "sens_Pw":    "R vs P_w (MPa)",
        "anim_title": "Near-wellbore Scale Accumulation M_NWZ(t)",
        "anim_years": "Modelling horizon (years)",
        "anim_btn":   "Show M(t)",
        "anim_yax":   "M_NWZ (kg)",
        "anim_xax":   "Time (years)",
        "rec_title": "Engineering Recommendations",
        "rec_none":  "✅ Scale risk is minimal. Routine monitoring recommended.",
        "det_title": "Calculation Details",
        "det_SI_e":  "SI_e (reservoir conditions)",
        "det_Sv":    "Specific Surface S_v (m²/m³)",
        "det_M_all": "Scale mass — full reservoir (kg)",
        "det_M_pbz": "Scale mass — NWZ (kg)",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CaCO₃ Scale Predictor",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0d0f14;
    color: #e8eaf0;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.8rem, 4vw, 3rem);
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e8eaf0 30%, #6c9cf5 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 0.88rem; color: #6b7280; font-weight: 300;
    letter-spacing: 0.02em; margin-bottom: 2rem;
}
.step-bar { display:flex; margin-bottom:1.8rem; border-radius:10px;
    overflow:hidden; border:1px solid #1e2230; }
.step-item { flex:1; padding:0.65rem 0.5rem; text-align:center; font-size:0.75rem;
    font-weight:600; color:#4b5563; background:#131620; letter-spacing:0.05em;
    text-transform:uppercase; border-right:1px solid #1e2230; transition:all 0.2s; }
.step-item:last-child { border-right:none; }
.step-item.active { background:#1a2640; color:#6c9cf5; border-bottom:2px solid #6c9cf5; }
.step-item.done   { background:#0f1f2e; color:#34d399; }
.card { background:#131620; border:1px solid #1e2230; border-radius:14px;
    padding:1.5rem 1.75rem; margin-bottom:0.8rem; }
.card h3 { font-family:'DM Serif Display',serif; font-size:1.1rem; color:#c9d1e0;
    margin-bottom:1rem; padding-bottom:0.4rem; border-bottom:1px solid #1e2230; }
.kpi-wrap { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
.kpi { background:#131620; border:1px solid #1e2230; border-radius:14px;
    padding:1.25rem 1.5rem; position:relative; overflow:hidden; }
.kpi::before { content:''; position:absolute; top:0;left:0;right:0;height:3px; }
.kpi.blue::before  { background:#3b82f6; }
.kpi.yellow::before{ background:#f59e0b; }
.kpi.orange::before{ background:#f97316; }
.kpi.red::before   { background:#ef4444; }
.kpi.green::before { background:#10b981; }
.kpi-lbl { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
    color:#6b7280; margin-bottom:0.4rem; }
.kpi-val { font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:600;
    line-height:1; margin-bottom:0.3rem; }
.kpi-unit { font-size:0.75rem; color:#6b7280; }
.badge { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
    font-size:0.7rem; font-weight:700; letter-spacing:0.04em; margin-top:0.3rem; }
.badge-blue   { background:rgba(59,130,246,.15);  color:#6c9cf5; }
.badge-yellow { background:rgba(245,158,11,.15);  color:#fbbf24; }
.badge-orange { background:rgba(249,115,22,.15);  color:#fb923c; }
.badge-red    { background:rgba(239,68,68,.15);   color:#f87171; }
.badge-green  { background:rgba(16,185,129,.15);  color:#34d399; }
.sec { font-family:'DM Serif Display',serif; font-size:1.2rem; color:#c9d1e0;
    margin:1.8rem 0 0.8rem; padding-bottom:0.35rem; border-bottom:1px solid #1e2230; }
.rec { background:#131620; border:1px solid #1e2230; border-radius:9px;
    padding:0.85rem 1.1rem; margin-bottom:0.5rem; font-size:0.88rem;
    color:#c9d1e0; line-height:1.55; }
.rec.info    { border-left:3px solid #6c9cf5; }
.rec.warning { border-left:3px solid #f59e0b; }
.rec.danger  { border-left:3px solid #ef4444; }
.rec.ok      { border-left:3px solid #10b981; }
.verr { background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.35);
    color:#fca5a5; border-radius:7px; padding:0.5rem 0.8rem;
    font-size:0.82rem; margin-top:0.2rem; }
div[data-testid="stNumberInput"]>div>input { background:#0d0f14!important;
    border-color:#1e2230!important; color:#e8eaf0!important; border-radius:7px!important; }
div[data-testid="stSelectbox"]>div { background:#0d0f14!important;
    border-color:#1e2230!important; color:#e8eaf0!important; border-radius:7px!important; }
div.stButton>button { background:#1a2640; color:#6c9cf5; border:1px solid #2d4a7a;
    border-radius:9px; padding:0.55rem 1.4rem; font-family:'DM Sans',sans-serif;
    font-weight:600; font-size:0.88rem; letter-spacing:0.02em; transition:all 0.2s; }
div.stButton>button:hover { background:#233457; border-color:#6c9cf5; }
div.stButton>button[kind="primary"] { background:linear-gradient(135deg,#1a3a6e,#1e4a94);
    color:#93b8ff; border:1px solid #3b6ab8; font-size:0.95rem; padding:0.7rem 2rem; }
[data-testid="stExpander"] { background:#131620; border:1px solid #1e2230; border-radius:11px; }
.stTabs [data-baseweb="tab-list"] { background:#131620; border-radius:9px; }
.stTabs [data-baseweb="tab"]      { color:#6b7280; }
.stTabs [aria-selected="true"]    { color:#6c9cf5!important; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════
_DEFAULTS = dict(
    lang="ru", step=0, results=None,
    T=60.0, pH=7.2,
    C_Ca=800.0, C_HCO3=400.0, C_NaK=5000.0, C_Cl=8000.0,
    m=0.20, rock_idx=0, k=50.0,
    Q_w=100.0, R_e=500.0, r_w=0.1, h=10.0,
    P_e=20.0, P_w=15.0, t=365.0, r_pbz=10.0,
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

L = LANG[st.session_state.lang]

# ═══════════════════════════════════════════════════════════════════════
# PLOTLY БАЗОВЫЙ СТИЛЬ
# ═══════════════════════════════════════════════════════════════════════
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1017",
    font=dict(family="DM Sans, sans-serif", color="#9ca3af", size=12),
    xaxis=dict(gridcolor="#1e2230", zerolinecolor="#2d3748", linecolor="#1e2230"),
    yaxis=dict(gridcolor="#1e2230", zerolinecolor="#2d3748", linecolor="#1e2230"),
    margin=dict(l=55, r=20, t=40, b=50),
    hoverlabel=dict(bgcolor="#1e2230", bordercolor="#2d3748",
                    font=dict(family="JetBrains Mono", color="#e8eaf0")),
)

# ═══════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════

def _si_style(si_max):
    """Возвращает (card_color, badge_class, label) по значению SI."""
    if si_max < 0:
        return "blue",   "badge-blue",   L["si_neg"]
    if si_max < 0.5:
        return "yellow", "badge-yellow", L["si_low"]
    if si_max < 1.5:
        return "orange", "badge-orange", L["si_mid"]
    return "red", "badge-red", L["si_high"]


def _r_style(r_max):
    if r_max < 1e-4:  return "green",  "#34d399"
    if r_max < 1e-2:  return "yellow", "#fbbf24"
    return "red", "#f87171"


def _m_style(m_pbz):
    if m_pbz < 10:   return "green",  "#34d399"
    if m_pbz < 500:  return "yellow", "#fbbf24"
    return "red", "#f87171"


def _num(label, key, lo, hi, step=None, help_txt=None, fmt="%.4g"):
    val = st.number_input(label, value=float(st.session_state[key]),
                          min_value=float(lo), max_value=float(hi),
                          step=step, help=help_txt, key=f"ni_{key}", format=fmt)
    st.session_state[key] = val
    return val


def _line(fig, x, y, color, name, xtitle, ytitle, fill=True, log_y=False):
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=2.5),
        fill="tozeroy" if fill else "none",
        fillcolor=color.replace("rgb","rgba").replace(")",",0.08)") if fill else None,
        hovertemplate=f"{xtitle}: %{{x:.2f}}<br>{ytitle}: %{{y:.4g}}<extra></extra>",
    ))
    fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle, **_PL)
    if log_y:
        fig.update_layout(yaxis_type="log")
    return fig


def _get_recommendations(SI_arr, R_arr, M_pbz, lang):
    recs = []
    SI_max = float(SI_arr.max())
    R_max  = float(R_arr.max())
    R_rw   = float(R_arr[0])
    s = st.session_state
    depr = s.P_e - s.P_w

    if SI_max <= 0:
        return [], "ok"

    ru = (lang == "ru")

    if SI_max > 1.5:
        recs.append(("danger",
            "🔴 SI > 1.5 — высокий риск нуклеации. Срочно рекомендуется применение ингибитора карбонатного солеотложения (фосфонаты, HEDP, полиакрилаты)." if ru else
            "🔴 SI > 1.5 — high nucleation risk. Immediate carbonate scale inhibitor treatment is recommended (phosphonates, HEDP, polyacrylates)."))
    elif SI_max > 0.5:
        recs.append(("warning",
            "🟡 SI > 0.5 — умеренное пересыщение. Рекомендуется периодический мониторинг химического состава воды и профилактическая дозировка ингибитора." if ru else
            "🟡 SI > 0.5 — moderate supersaturation. Periodic water chemistry monitoring and preventive inhibitor dosing recommended."))

    if R_rw > R_arr.mean() * 2:
        recs.append(("danger",
            "⚠️ Максимальный риск сконцентрирован в призабойной зоне. Первоочередная мера — закачка ингибитора в скважину (squeeze treatment)." if ru else
            "⚠️ Maximum risk concentrated near the wellbore. Priority action: inhibitor squeeze treatment into the well."))

    if depr > 7.0:
        recs.append(("warning",
            f"🔩 Депрессия ΔP = {depr:.1f} МПа — высокая. Снижение депрессии уменьшит падение давления у забоя и замедлит десорбцию CO₂, снижая риск осаждения." if ru else
            f"🔩 Drawdown ΔP = {depr:.1f} MPa is high. Reducing drawdown will slow CO₂ degassing near the wellbore and decrease scale risk."))

    if s.pH > 7.5:
        recs.append(("warning",
            f"⚗️ pH = {s.pH:.1f} — щелочная среда усиливает осаждение CaCO₃. Контроль pH (подкисление) может снизить SI." if ru else
            f"⚗️ pH = {s.pH:.1f} — alkaline conditions promote CaCO₃ precipitation. pH adjustment (acidification) may reduce SI."))
    elif s.pH < 6.5:
        recs.append(("info",
            f"⚗️ pH = {s.pH:.1f} — кислая среда снижает риск осаждения кальцита. Однако возможна коррозия оборудования." if ru else
            f"⚗️ pH = {s.pH:.1f} — acidic conditions reduce calcite precipitation risk. However, equipment corrosion may be a concern."))

    if s.T > 80:
        recs.append(("warning",
            f"🌡️ T = {s.T:.0f} °C — высокая температура ускоряет кристаллизацию. Рассмотрите теплоизоляцию НКТ или термостойкий ингибитор." if ru else
            f"🌡️ T = {s.T:.0f} °C — high temperature accelerates crystallisation. Consider tubing insulation or a heat-stable inhibitor."))

    if M_pbz > 500:
        recs.append(("danger",
            f"💾 Ожидаемая масса осадка в ПЗП: {M_pbz:.0f} кг за период. Необходимо планирование химической обработки скважины." if ru else
            f"💾 Expected near-wellbore scale mass: {M_pbz:.0f} kg over the period. Well chemical treatment should be planned."))
    elif M_pbz > 10:
        recs.append(("warning",
            f"💾 Масса осадка в ПЗП: {M_pbz:.1f} кг. Рекомендуется профилактическая обработка ингибитором." if ru else
            f"💾 Near-wellbore scale mass: {M_pbz:.1f} kg. Preventive inhibitor treatment is advisable."))

    if not recs:
        return [], "ok"
    severity = "danger" if any(c == "danger" for c, _ in recs) else "warning"
    return recs, severity


def _heatmap_fig(r_arr, R_arr, L):
    """Радиальная тепловая карта на Barpolar. Лёгкая, без утечек памяти."""
    MAX_RINGS = 40
    step = max(1, len(r_arr) // MAX_RINGS)
    r_ds = r_arr[::step]
    R_ds = R_arr[::step]
    R_mx = float(R_ds.max()) if R_ds.max() > 0 else 1.0
    widths = np.diff(r_ds, prepend=0.0)

    def _clr(t):
        t = max(0., min(1., t))
        if t < 0.33:
            tt = t/0.33
            return f"rgba({int(30-4*tt)},{int(58+49*tt)},{int(95-21*tt)},0.92)"
        elif t < 0.66:
            tt = (t-0.33)/0.33
            return f"rgba({int(26+154*tt)},{int(107-24*tt)},{int(74-65*tt)},0.92)"
        else:
            tt = (t-0.66)/0.34
            return f"rgba({int(180+59*tt)},{int(83-15*tt)},{int(9+59*tt)},0.92)"

    N, dth = 36, 10.0
    thetas = np.arange(0, 360, dth)
    fig = go.Figure()
    for r_v, R_v, w in zip(r_ds, R_ds, widths):
        clr = _clr(R_v / R_mx)
        fig.add_trace(go.Barpolar(
            r=[r_v]*N, theta=thetas, width=[dth]*N, base=[r_v-w]*N,
            marker=dict(color=clr, line=dict(width=0)),
            hovertemplate=f"r = {r_v:.1f} m<br>R = {R_v:.3e} {L['kpi_R_unit']}<extra></extra>",
            showlegend=False,
        ))
    fig.add_trace(go.Scatterpolar(
        r=[0.], theta=[0], mode="markers",
        marker=dict(size=11, color="white", symbol="circle"),
        hovertemplate=f"{L['wellbore']}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1017",
            radialaxis=dict(visible=True, showgrid=True, gridcolor="#1e2230",
                tickfont=dict(color="#6b7280", size=10, family="JetBrains Mono"),
                ticksuffix=" m", range=[0, float(r_ds[-1])*1.05]),
            angularaxis=dict(showgrid=False, showticklabels=False),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#9ca3af"),
        height=460, margin=dict(l=40, r=40, t=30, b=30),
        hoverlabel=dict(bgcolor="#1e2230", bordercolor="#2d3748",
                        font=dict(family="JetBrains Mono", color="#e8eaf0")),
    )
    return fig


def _mass_fast(r_arr, R_arr, h, t):
    return t * float(np.trapezoid(R_arr * 2 * math.pi * r_arr * h, r_arr))

# ═══════════════════════════════════════════════════════════════════════
# ШАПКА
# ═══════════════════════════════════════════════════════════════════════
col_h, col_lng = st.columns([5, 1])
with col_h:
    st.markdown('<div class="hero-title">CaCO₃ Scale Predictor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{L["subtitle"]}</div>', unsafe_allow_html=True)
with col_lng:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button(L["lang_btn"], key="lang_toggle"):
        st.session_state.lang = "en" if st.session_state.lang == "ru" else "ru"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# ПРОГРЕСС-БАР
# ═══════════════════════════════════════════════════════════════════════
step = st.session_state.step
steps_labels = [L["step1"], L["step2"], L["step3"]]
bar_html = '<div class="step-bar">'
for i, lbl in enumerate(steps_labels):
    if i < step:
        cls, pfx = "done", "✓ "
    elif i == step:
        cls, pfx = "active", f"{i+1}. "
    else:
        cls, pfx = "", f"{i+1}. "
    bar_html += f'<div class="step-item {cls}">{pfx}{lbl}</div>'
bar_html += "</div>"
st.markdown(bar_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# ШАГ 0 — ХИМИЯ ВОДЫ
# ═══════════════════════════════════════════════════════════════════════
errors = []

if step == 0:
    st.markdown('<div class="card"><h3>🧪 ' + L["step1"] + '</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        T_v = _num(L["T_lbl"],    "T",     20.0, 120.0, 1.0,  L["T_hlp"])
        if not (20 <= T_v <= 120):
            st.markdown(f'<div class="verr">{L["err_T"]}</div>', unsafe_allow_html=True); errors.append(1)
        pH_v = _num(L["pH_lbl"],  "pH",    1.0,  14.0,  0.1,  L["pH_hlp"])
        if not (1 <= pH_v <= 14):
            st.markdown(f'<div class="verr">{L["err_pH"]}</div>', unsafe_allow_html=True); errors.append(1)
        _num(L["CCa_lbl"],  "C_Ca",   0.0, 100000.0, 10.0)
    with c2:
        _num(L["CHCO3_lbl"],"C_HCO3", 0.0, 100000.0, 10.0)
        _num(L["CNaK_lbl"], "C_NaK",  0.0, 200000.0, 50.0)
        _num(L["CCl_lbl"],  "C_Cl",   0.0, 300000.0, 50.0)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(L["next"], disabled=bool(errors)):
        st.session_state.step = 1; st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# ШАГ 1 — ФЕС КОЛЛЕКТОРА
# ═══════════════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown('<div class="card"><h3>🪨 ' + L["step2"] + '</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        m_v = _num(L["m_lbl"], "m", 0.01, 0.99, 0.01, L["m_hlp"])
        if not (0 < m_v < 1):
            st.markdown(f'<div class="verr">{L["err_m"]}</div>', unsafe_allow_html=True); errors.append(1)
        rock_i = st.selectbox(L["rock_lbl"],
            options=range(len(L["rock_opt"])),
            format_func=lambda i: L["rock_opt"][i],
            index=st.session_state.rock_idx, key="ni_rock")
        st.session_state.rock_idx = rock_i
    with c2:
        _num(L["k_lbl"], "k", 0.01, 10000.0, 1.0)
    st.markdown("</div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button(L["back"]): st.session_state.step = 0; st.rerun()
    with b2:
        if st.button(L["next"], disabled=bool(errors)): st.session_state.step = 2; st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# ШАГ 2 — ГИДРОДИНАМИКА
# ═══════════════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown('<div class="card"><h3>⛽ ' + L["step3"] + '</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        _num(L["Qw_lbl"], "Q_w", 0.1,   5000.0, 10.0)
        _num(L["Re_lbl"], "R_e", 10.0,  5000.0, 10.0)
        _num(L["rw_lbl"], "r_w", 0.01,  1.0,    0.01)
        _num(L["h_lbl"],  "h",   0.5,   500.0,  0.5)
        _num(L["rpbz_lbl"],"r_pbz", 1.0, 100.0, 1.0)
    with c2:
        Pe_v = _num(L["Pe_lbl"], "P_e", 0.5, 100.0, 0.5)
        Pw_v = _num(L["Pw_lbl"], "P_w", 0.1, 100.0, 0.5)
        if Pw_v >= Pe_v:
            st.markdown(f'<div class="verr">{L["err_Pw"]}</div>', unsafe_allow_html=True); errors.append(1)
        else:
            depr = Pe_v - Pw_v
            tag = "ru" if st.session_state.lang == "ru" else "en"
            st.markdown(f"**{L['depr_lbl']}:** `{depr:.2f} {'МПа' if tag=='ru' else 'MPa'}`")

        if st.session_state.r_w >= st.session_state.R_e:
            st.markdown(f'<div class="verr">{L["err_rw"]}</div>', unsafe_allow_html=True); errors.append(1)

        _num(L["t_lbl"], "t", 1.0, 36500.0, 30.0)
    st.markdown("</div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button(L["back"]): st.session_state.step = 1; st.rerun()
    with b2:
        if st.button(L["calc"], type="primary", disabled=bool(errors)):
            with st.spinner("Расчёт..." if st.session_state.lang == "ru" else "Calculating..."):
                s = st.session_state
                res = run_model(
                    T=s.T, pH=s.pH, C_Ca=s.C_Ca, C_HCO3=s.C_HCO3,
                    C_NaK=s.C_NaK, C_Cl=s.C_Cl,
                    m=s.m, rock_type=L["rock_val"][s.rock_idx], k=s.k,
                    Q_w=s.Q_w, R_e=s.R_e, r_w=s.r_w, h=s.h,
                    P_e=s.P_e, P_w=s.P_w, t=s.t,
                    n_points=150, r_pbz=s.r_pbz,
                )
            st.session_state.results = res
            st.session_state.step = 3
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# ШАГ 3 — РЕЗУЛЬТАТЫ
# ═══════════════════════════════════════════════════════════════════════
elif step == 3:
    res    = st.session_state.results
    r_arr  = res["r_arr"]
    SI_arr = res["SI_arr"]
    R_arr  = res["R_arr"]
    P_arr  = res["P_arr"]
    v_arr  = res["v_arr"]
    M_pbz  = res["M_pbz"]
    SI_max = float(SI_arr.max())
    R_max  = float(R_arr.max())
    s      = st.session_state

    if st.button(L["edit"]):
        st.session_state.step = 0; st.rerun()

    # ── KPI ────────────────────────────────────────────────────────────
    si_col, si_badge, si_lbl = _si_style(SI_max)
    r_col,  r_clr           = _r_style(R_max)
    m_col,  m_clr           = _m_style(M_pbz)

    si_clr_map = {"blue":"#6c9cf5","yellow":"#fbbf24","orange":"#fb923c","red":"#f87171","green":"#34d399"}
    si_val_clr = si_clr_map[si_col]

    st.markdown(f"""
    <div class="kpi-wrap">
      <div class="kpi {si_col}">
        <div class="kpi-lbl">{L["kpi_SI_lbl"]}</div>
        <div class="kpi-val" style="color:{si_val_clr}">{SI_max:+.3f}</div>
        <span class="badge {si_badge}">{si_lbl}</span>
      </div>
      <div class="kpi {r_col}">
        <div class="kpi-lbl">{L["kpi_R_lbl"]}</div>
        <div class="kpi-val" style="color:{r_clr}">{R_max:.3e}</div>
        <div class="kpi-unit">{L["kpi_R_unit"]}</div>
      </div>
      <div class="kpi {m_col}">
        <div class="kpi-lbl">{L["kpi_M_lbl"]}</div>
        <div class="kpi-val" style="color:{m_clr}">{M_pbz:.1f}</div>
        <div class="kpi-unit">{L["kpi_M_unit"]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ГРАФИКИ ────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec">📈 {"Профили параметров" if s.lang=="ru" else "Parameter Profiles"}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([L["tab_SI"], L["tab_P"], L["tab_v"], L["tab_R"]])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r_arr, y=SI_arr, mode="lines",
            line=dict(color="#6c9cf5", width=2.5),
            fill="tozeroy", fillcolor="rgba(108,156,245,0.08)",
            hovertemplate=f"{L['r_ax']}: %{{x:.2f}}<br>SI: %{{y:.4f}}<extra></extra>",
        ))
        fig.add_hline(y=0, line=dict(color="#ef4444", width=1.5, dash="dash"),
                      annotation_text="SI = 0", annotation_font_color="#ef4444")
        fig.update_layout(xaxis_title=L["r_ax"], yaxis_title=L["SI_ax"],
                          title=L["tab_SI"], **_PL)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=r_arr, y=P_arr, mode="lines",
            line=dict(color="#a78bfa", width=2.5),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
            hovertemplate=f"{L['r_ax']}: %{{x:.2f}}<br>P: %{{y:.3f}}<extra></extra>",
        ))
        fig2.update_layout(xaxis_title=L["r_ax"], yaxis_title=L["P_ax"],
                           title=L["tab_P"], **_PL)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=r_arr, y=v_arr, mode="lines",
            line=dict(color="#34d399", width=2.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
            hovertemplate=f"{L['r_ax']}: %{{x:.2f}}<br>v: %{{y:.3e}}<extra></extra>",
        ))
        fig3.update_layout(xaxis_title=L["r_ax"], yaxis_title=L["v_ax"],
                           title=L["tab_v"], yaxis_type="log", **_PL)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=r_arr, y=R_arr, mode="lines",
            line=dict(color="#f87171", width=2.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.08)",
            hovertemplate=f"{L['r_ax']}: %{{x:.2f}}<br>R: %{{y:.4e}}<extra></extra>",
        ))
        # Зона ПЗП
        pbz_mask = r_arr <= s.r_pbz
        if pbz_mask.sum() >= 2:
            fig4.add_vrect(x0=float(r_arr[0]), x1=float(s.r_pbz),
                           fillcolor="rgba(249,115,22,0.07)",
                           line=dict(color="#f97316", width=1, dash="dot"),
                           annotation_text="ПЗП" if s.lang=="ru" else "NWZ",
                           annotation_font_color="#f97316")
        fig4.update_layout(xaxis_title=L["r_ax"], yaxis_title=L["R_ax"],
                           title=L["tab_R"], **_PL)
        st.plotly_chart(fig4, use_container_width=True)

    # ── ТЕПЛОВАЯ КАРТА ─────────────────────────────────────────────────
    st.markdown(f'<div class="sec">🗺️ {L["heatmap_title"]}</div>', unsafe_allow_html=True)
    st.caption(L["heatmap_cap"])
    st.plotly_chart(_heatmap_fig(r_arr, R_arr, L), use_container_width=True)

    # ── АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ────────────────────────────────────────
    st.markdown(f'<div class="sec">🔬 {L["sens_title"]}</div>', unsafe_allow_html=True)
    st.caption(L["sens_note"])

    if st.button(L["sens_btn"], key="run_sens"):
        with st.spinner(L["sens_run"]):
            base = dict(
                T=s.T, pH=s.pH, C_Ca=s.C_Ca, C_HCO3=s.C_HCO3,
                C_NaK=s.C_NaK, C_Cl=s.C_Cl,
                m=s.m, rock_type=L["rock_val"][s.rock_idx], k=s.k,
                Q_w=s.Q_w, R_e=s.R_e, r_w=s.r_w, h=s.h,
                P_e=s.P_e, P_w=s.P_w, t=s.t, n_points=80, r_pbz=s.r_pbz,
            )

            def _sweep(param, vals, guard=None):
                out = []
                for v in vals:
                    kw = {**base, param: v}
                    if guard and not guard(kw):
                        out.append(float("nan")); continue
                    try:
                        r2 = run_model(**kw)
                        out.append(float(r2["R_arr"][0]))
                    except Exception:
                        out.append(float("nan"))
                return out

            pH_rng = np.linspace(5.5, 9.5, 35)
            T_rng  = np.linspace(20,  120,  35)
            Pw_rng = np.linspace(s.P_w * 0.3, s.P_e * 0.97, 35)

            R_pH = _sweep("pH",  pH_rng)
            R_T  = _sweep("T",   T_rng)
            R_Pw = _sweep("P_w", Pw_rng,
                          guard=lambda kw: kw["P_w"] < kw["P_e"])

        colors = ["#6c9cf5", "#a78bfa", "#34d399"]
        xlabels = ["pH", "T (°C)", "P_w (MPa)" if s.lang=="en" else "P_w (МПа)"]
        titles  = [L["sens_pH"], L["sens_T"], L["sens_Pw"]]
        data    = [(pH_rng, R_pH), (T_rng, R_T), (Pw_rng, R_Pw)]

        fig_s = make_subplots(rows=1, cols=3, subplot_titles=titles,
                              horizontal_spacing=0.1)
        for col_i, ((xd, yd), clr, xl) in enumerate(zip(data, colors, xlabels), 1):
            fig_s.add_trace(go.Scatter(
                x=xd, y=yd, mode="lines",
                line=dict(color=clr, width=2.5),
                fill="tozeroy", fillcolor=clr.replace("#","rgba(")+"1)" ,
                hovertemplate=f"{xl}: %{{x:.2f}}<br>R: %{{y:.3e}}<extra></extra>",
                showlegend=False,
            ), row=1, col=col_i)

        base_layout = {k: v for k, v in _PL.items() if k not in ("xaxis","yaxis")}
        fig_s.update_layout(height=360, showlegend=False, **base_layout)
        for ax in ["xaxis","yaxis","xaxis2","yaxis2","xaxis3","yaxis3"]:
            fig_s.update_layout(**{ax: dict(gridcolor="#1e2230", linecolor="#1e2230",
                                            zerolinecolor="#2d3748", color="#9ca3af")})
        st.plotly_chart(fig_s, use_container_width=True)

    # ── АНИМАЦИЯ M(t) ──────────────────────────────────────────────────
    st.markdown(f'<div class="sec">⏱ {L["anim_title"]}</div>', unsafe_allow_html=True)
    years = st.slider(L["anim_years"], 1, 20, 5, key="anim_sl")
    if st.button(L["anim_btn"], key="anim_btn"):
        t_pts = np.linspace(0, years * 365, 80)
        pbz_mask = r_arr <= s.r_pbz
        r_pbz_arr = r_arr[pbz_mask] if pbz_mask.sum() >= 2 else r_arr[:2]
        R_pbz_arr = R_arr[pbz_mask] if pbz_mask.sum() >= 2 else R_arr[:2]
        M_pts = [_mass_fast(r_pbz_arr, R_pbz_arr, s.h, t_i) for t_i in t_pts]

        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(
            x=t_pts / 365, y=M_pts, mode="lines",
            line=dict(color="#f59e0b", width=3),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.10)",
            hovertemplate=f"t: %{{x:.2f}} {'лет' if s.lang=='ru' else 'yr'}<br>M: %{{y:.1f}} кг<extra></extra>",
        ))
        fig_a.update_layout(
            xaxis_title=L["anim_xax"], yaxis_title=L["anim_yax"],
            title=L["anim_title"], **_PL,
        )
        st.plotly_chart(fig_a, use_container_width=True)

    # ── РЕКОМЕНДАЦИИ ───────────────────────────────────────────────────
    st.markdown(f'<div class="sec">💡 {L["rec_title"]}</div>', unsafe_allow_html=True)
    recs, _ = _get_recommendations(SI_arr, R_arr, M_pbz, s.lang)
    if not recs:
        st.markdown(f'<div class="rec ok">{L["rec_none"]}</div>', unsafe_allow_html=True)
    else:
        for cls, txt in recs:
            st.markdown(f'<div class="rec {cls}">{txt}</div>', unsafe_allow_html=True)

    # ── ДЕТАЛИ ─────────────────────────────────────────────────────────
    with st.expander(L["det_title"]):
        d1, d2 = st.columns(2)
        with d1:
            st.metric(L["det_SI_e"],  f"{res['SI_e']:+.4f}")
            st.metric(L["det_Sv"],    f"{res['Sv']:.0f}")
            st.metric("SI_max",       f"{SI_max:+.4f}")
        with d2:
            st.metric("R_max",        f"{R_max:.4e} {L['kpi_R_unit']}")
            st.metric(L["det_M_pbz"], f"{M_pbz:.2f} {L['kpi_M_unit']}")
            st.metric(L["det_M_all"], f"{res['M_t']:.1f} {L['kpi_M_unit']}")

