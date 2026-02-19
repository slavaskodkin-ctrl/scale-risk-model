"""
CaCO₃ Scale Deposition Predictor
Streamlit MVP — инженерный калькулятор риска карбонатного солеотложения.
"""

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from model import run_model

# ─────────────────────────────────────────────────────────────────────────────
# LOCALIZATION
# ─────────────────────────────────────────────────────────────────────────────
LANG = {
    "ru": {
        "app_title": "CaCO₃ Scale Predictor",
        "app_subtitle": "Прогнозирование риска карбонатного солеотложения в поровом пространстве коллектора",
        "lang_toggle": "🇬🇧 English",
        # Steps
        "step1": "Химия воды",
        "step2": "ФЕС коллектора",
        "step3": "Гидродинамика",
        "step_next": "Далее →",
        "step_back": "← Назад",
        "step_calc": "▶  Рассчитать",
        "step_label": "Шаг",
        "of": "из",
        # Inputs — chem
        "T_label": "Температура пласта (°C)",
        "T_help": "Допустимый диапазон: 20–120 °C",
        "pH_label": "pH пластовой воды",
        "pH_help": "Допустимый диапазон: 1–14",
        "CCa_label": "Концентрация Ca²⁺ (мг/л)",
        "CHCO3_label": "Концентрация HCO₃⁻ (мг/л)",
        "CNaK_label": "Концентрация Na⁺ + K⁺ (мг/л)",
        "CCl_label": "Концентрация Cl⁻ (мг/л)",
        # Inputs — reservoir
        "m_label": "Пористость (доли)",
        "m_help": "Допустимый диапазон: 0–1",
        "rock_label": "Тип коллектора",
        "rock_options": ["Песчаник", "Карбонат"],
        "rock_values": ["sandstone", "carbonate"],
        "k_label": "Проницаемость по воде (мД)",
        # Inputs — hydro
        "Qw_label": "Дебит воды (м³/сут)",
        "Re_label": "Радиус контура питания Rₑ (м)",
        "rw_label": "Радиус скважины rw (м)",
        "h_label": "Эффективная толщина пласта h (м)",
        "Pe_label": "Пластовое давление Pₑ (МПа)",
        "Pw_label": "Забойное давление Pw (МПа)",
        "t_label": "Период эксплуатации t (сут)",
        "depression_label": "Депрессия ΔP = Pₑ − Pw (МПа)",
        # Results
        "results_title": "Результаты расчёта",
        "kpi_SI": "Индекс насыщения SI(r)",
        "kpi_SI_max": "Максимальный SI",
        "kpi_R_max": "Макс. локальный риск R(r)",
        "kpi_M": "Масса осадка за период",
        "kpi_M_unit": "тонн",
        "kpi_R_unit": "т/(м³·сут)",
        "si_under": "Недонасыщено",
        "si_near": "Близко к равновесию",
        "si_over": "Пересыщено — риск!",
        # Charts
        "chart_SI": "Индекс насыщения SI(r)",
        "chart_P": "Давление P(r)",
        "chart_v": "Поровая скорость v(r)",
        "chart_R": "Скорость солеотложения R(r)",
        "chart_heatmap": "Тепловая карта зоны риска",
        "heatmap_note": "Радиальная схема пласта. Центр — скважина. Заливка по R(r).",
        "r_axis": "Радиус r (м)",
        "SI_axis": "SI(r)",
        "P_axis": "P (МПа)",
        "v_axis": "v (м/с)",
        "R_axis": "R (т/(м³·сут))",
        # Sensitivity
        "sens_title": "Анализ чувствительности",
        "sens_btn": "Запустить анализ чувствительности",
        "sens_pH": "R(r_w) vs pH",
        "sens_T": "R(r_w) vs Температура",
        "sens_Pw": "R(r_w) vs Забойное давление",
        "sens_note": "Оценка ведётся по значению R у скважины (r = rw), остальные параметры — как введены.",
        # Recommendations
        "rec_title": "Инженерные рекомендации",
        "rec_none": "✅ Риск солеотложения отсутствует. Текущие условия благоприятны.",
        # Animation
        "anim_title": "Накопление осадка во времени",
        "anim_btn": "Показать анимацию M(t)",
        "anim_years": "Горизонт моделирования (лет)",
        # Errors
        "err_T": "⛔ Температура должна быть в диапазоне 20–120 °C",
        "err_pH": "⛔ pH должен быть в диапазоне 1–14",
        "err_m": "⛔ Пористость должна быть в диапазоне 0–1 (не включая границы)",
        "err_Pw": "⛔ Забойное давление должно быть меньше пластового",
        "err_rw": "⛔ Радиус скважины должен быть меньше радиуса контура питания",
        "details_expander": "Детали расчёта (промежуточные параметры)",
    },
    "en": {
        "app_title": "CaCO₃ Scale Predictor",
        "app_subtitle": "Carbonate scale deposition risk prediction in reservoir pore space",
        "lang_toggle": "🇷🇺 Русский",
        "step1": "Water Chemistry",
        "step2": "Reservoir Properties",
        "step3": "Hydrodynamics",
        "step_next": "Next →",
        "step_back": "← Back",
        "step_calc": "▶  Calculate",
        "step_label": "Step",
        "of": "of",
        "T_label": "Reservoir Temperature (°C)",
        "T_help": "Valid range: 20–120 °C",
        "pH_label": "Formation Water pH",
        "pH_help": "Valid range: 1–14",
        "CCa_label": "Ca²⁺ Concentration (mg/L)",
        "CHCO3_label": "HCO₃⁻ Concentration (mg/L)",
        "CNaK_label": "Na⁺ + K⁺ Concentration (mg/L)",
        "CCl_label": "Cl⁻ Concentration (mg/L)",
        "m_label": "Porosity (fraction)",
        "m_help": "Valid range: 0–1",
        "rock_label": "Reservoir Type",
        "rock_options": ["Sandstone", "Carbonate"],
        "rock_values": ["sandstone", "carbonate"],
        "k_label": "Water Permeability (mD)",
        "Qw_label": "Water Flow Rate (m³/day)",
        "Re_label": "Drainage Radius Rₑ (m)",
        "rw_label": "Wellbore Radius rw (m)",
        "h_label": "Net Pay Thickness h (m)",
        "Pe_label": "Reservoir Pressure Pₑ (MPa)",
        "Pw_label": "Bottom-hole Pressure Pw (MPa)",
        "t_label": "Production Period t (days)",
        "depression_label": "Drawdown ΔP = Pₑ − Pw (MPa)",
        "results_title": "Calculation Results",
        "kpi_SI": "Saturation Index SI(r)",
        "kpi_SI_max": "Maximum SI",
        "kpi_R_max": "Max Local Risk R(r)",
        "kpi_M": "Scale Mass over Period",
        "kpi_M_unit": "tonnes",
        "kpi_R_unit": "t/(m³·day)",
        "si_under": "Undersaturated",
        "si_near": "Near equilibrium",
        "si_over": "Supersaturated — risk!",
        "chart_SI": "Saturation Index SI(r)",
        "chart_P": "Pressure Profile P(r)",
        "chart_v": "Pore Velocity v(r)",
        "chart_R": "Deposition Rate R(r)",
        "chart_heatmap": "Risk Zone Heatmap",
        "heatmap_note": "Radial reservoir cross-section. Center — wellbore. Color fill by R(r).",
        "r_axis": "Radius r (m)",
        "SI_axis": "SI(r)",
        "P_axis": "P (MPa)",
        "v_axis": "v (m/s)",
        "R_axis": "R (t/(m³·day))",
        "sens_title": "Sensitivity Analysis",
        "sens_btn": "Run Sensitivity Analysis",
        "sens_pH": "R(r_w) vs pH",
        "sens_T": "R(r_w) vs Temperature",
        "sens_Pw": "R(r_w) vs Bottom-hole Pressure",
        "sens_note": "R is evaluated at the wellbore (r = rw). All other parameters as entered.",
        "rec_title": "Engineering Recommendations",
        "rec_none": "✅ No scale risk detected. Current conditions are favourable.",
        "anim_title": "Scale Accumulation Over Time",
        "anim_btn": "Show M(t) Animation",
        "anim_years": "Modelling horizon (years)",
        "err_T": "⛔ Temperature must be between 20 and 120 °C",
        "err_pH": "⛔ pH must be between 1 and 14",
        "err_m": "⛔ Porosity must be between 0 and 1 (exclusive)",
        "err_Pw": "⛔ Bottom-hole pressure must be less than reservoir pressure",
        "err_rw": "⛔ Wellbore radius must be less than drainage radius",
        "details_expander": "Calculation details (intermediate parameters)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CaCO₃ Scale Predictor",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0d0f14;
    color: #e8eaf0;
}

/* ── Main header ── */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e8eaf0 30%, #6c9cf5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 2.5rem;
}

/* ── Step wizard ── */
.step-bar {
    display: flex;
    gap: 0;
    margin-bottom: 2rem;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1e2230;
}
.step-item {
    flex: 1;
    padding: 0.75rem 1rem;
    text-align: center;
    font-size: 0.8rem;
    font-weight: 500;
    color: #4b5563;
    background: #131620;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-right: 1px solid #1e2230;
    transition: all 0.25s;
}
.step-item:last-child { border-right: none; }
.step-item.active {
    background: #1a2640;
    color: #6c9cf5;
    border-bottom: 2px solid #6c9cf5;
}
.step-item.done {
    background: #131f2e;
    color: #34d399;
}

/* ── Input card ── */
.input-card {
    background: #131620;
    border: 1px solid #1e2230;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1rem;
}
.input-card h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #c9d1e0;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2230;
}

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: #131620;
    border: 1px solid #1e2230;
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.blue::before  { background: #3b82f6; }
.kpi-card.yellow::before { background: #f59e0b; }
.kpi-card.red::before   { background: #ef4444; }
.kpi-card.green::before { background: #10b981; }
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #6b7280;
}
.kpi-badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-top: 0.4rem;
}
.badge-blue   { background: rgba(59,130,246,0.15); color: #6c9cf5; }
.badge-yellow { background: rgba(245,158,11,0.15); color: #fbbf24; }
.badge-red    { background: rgba(239,68,68,0.15);  color: #f87171; }
.badge-green  { background: rgba(16,185,129,0.15); color: #34d399; }

/* ── Section headers ── */
.section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #c9d1e0;
    margin: 2rem 0 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2230;
}

/* ── Recommendation cards ── */
.rec-card {
    background: #131620;
    border: 1px solid #1e2230;
    border-left: 3px solid #6c9cf5;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    color: #c9d1e0;
    line-height: 1.5;
}
.rec-card.warning { border-left-color: #f59e0b; }
.rec-card.danger  { border-left-color: #ef4444; }
.rec-card.ok      { border-left-color: #10b981; }

/* ── Validation error ── */
.val-error {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.4);
    color: #fca5a5;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* ── Streamlit overrides ── */
div[data-testid="stNumberInput"] > div > input,
div[data-testid="stSelectbox"] > div {
    background: #0d0f14 !important;
    border-color: #1e2230 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}
div.stButton > button {
    background: #1a2640;
    color: #6c9cf5;
    border: 1px solid #2d4a7a;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: #233457;
    border-color: #6c9cf5;
    color: #93b8ff;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a6e, #1e4a94);
    color: #93b8ff;
    border: 1px solid #3b6ab8;
    font-size: 1rem;
    padding: 0.75rem 2rem;
}
.stSlider > div > div { background: #1e2230 !important; }
[data-testid="stExpander"] {
    background: #131620;
    border: 1px solid #1e2230;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] { background: #131620; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: #6b7280; }
.stTabs [aria-selected="true"] { color: #6c9cf5 !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        lang="ru", step=0, results=None,
        T=60.0, pH=7.2,
        C_Ca=800.0, C_HCO3=400.0, C_NaK=5000.0, C_Cl=8000.0,
        m=0.20, rock_idx=0, k=50.0,
        Q_w=100.0, R_e=500.0, r_w=0.1, h=10.0,
        P_e=20.0, P_w=15.0, t=365.0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
L = LANG[st.session_state.lang]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1017",
    font=dict(family="DM Sans, sans-serif", color="#9ca3af", size=12),
    xaxis=dict(gridcolor="#1e2230", zerolinecolor="#2d3748", linecolor="#1e2230"),
    yaxis=dict(gridcolor="#1e2230", zerolinecolor="#2d3748", linecolor="#1e2230"),
    margin=dict(l=50, r=20, t=40, b=50),
    hoverlabel=dict(bgcolor="#1e2230", bordercolor="#2d3748",
                    font=dict(family="JetBrains Mono", color="#e8eaf0")),
)

def _plotly_line(x, y, name, color, xlabel, ylabel, title, fill=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=2.5),
        fill="tozeroy" if fill else "none",
        fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba") if fill else None,
        hovertemplate=f"{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.4g}}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#c9d1e0")),
                      xaxis_title=xlabel, yaxis_title=ylabel, **PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helper used in animation (avoid reimporting)
# ─────────────────────────────────────────────────────────────────────────────
def _calc_total_mass_fast(r_arr, R_arr, h, t):
    integrand = R_arr * 2 * math.pi * r_arr * h
    return t * np.trapezoid(integrand, r_arr)

def _si_color(si_max):
    if si_max < 0:   return "blue",   "badge-blue"
    if si_max < 0.5: return "yellow", "badge-yellow"
    return "red", "badge-red"

def _get_recommendations(r_arr, SI_arr, R_arr, M_t, res, lang):
    """Generate engineering recommendations from results."""
    recs = []
    SI_max  = float(SI_arr.max())
    R_max   = float(R_arr.max())
    R_at_rw = float(R_arr[0])
    R_at_Re = float(R_arr[-1])

    if SI_max <= 0:
        return [], "ok"

    severity = "warning" if SI_max < 1.0 else "danger"

    if lang == "ru":
        if SI_max > 0.5:
            recs.append(("warning", "📊 SI > 0.5 — раствор пересыщен. Рекомендуется регулярный мониторинг химического состава воды."))
        if SI_max > 1.0:
            recs.append(("danger",  "🧪 SI > 1.0 — высокий риск нуклеации. Рассмотрите применение ингибитора карбонатного солеотложения (фосфонаты, полиакрилаты)."))
        if R_at_rw > R_at_Re * 0.5:
            recs.append(("danger",  "⚠️ Максимальный риск сконцентрирован в призабойной зоне. Первостепенная защита — обработка скважины."))
        if R_at_Re > R_at_rw * 0.5 and R_at_rw < R_at_Re:
            recs.append(("warning", "📍 Риск выше на периферии пласта. Рассмотрите закачку ингибитора через нагнетательную систему."))
        Pe = st.session_state.P_e
        Pw = st.session_state.P_w
        depression = Pe - Pw
        if depression > 5.0:
            recs.append(("warning", f"🔩 Депрессия ΔP = {depression:.1f} МПа высокая. Снижение депрессии уменьшит пересыщение и скорость отложений."))
        pH_val = st.session_state.pH
        if pH_val < 7.0:
            recs.append(("warning", "⚗️ pH < 7 — кислая среда. Контроль pH может снизить концентрацию CO₃²⁻ и риск."))
        if pH_val > 8.0:
            recs.append(("danger",  "⚗️ pH > 8 — щелочная среда усиливает риск карбонатного осадка. Рассмотрите подкисление воды."))
        if M_t > 1.0:
            recs.append(("danger",  f"💾 Ожидаемая масса осадка за период: {M_t:.2f} т. Планируйте химическую обработку скважины."))
        T_val = st.session_state.T
        if T_val > 80:
            recs.append(("warning", "🌡️ Высокая температура ускоряет кристаллизацию. Теплоизоляция НКТ может снизить риск."))
    else:
        if SI_max > 0.5:
            recs.append(("warning", "📊 SI > 0.5 — solution is supersaturated. Regular water chemistry monitoring is recommended."))
        if SI_max > 1.0:
            recs.append(("danger",  "🧪 SI > 1.0 — high nucleation risk. Consider carbonate scale inhibitor treatment (phosphonates, polyacrylates)."))
        if R_at_rw > R_at_Re * 0.5:
            recs.append(("danger",  "⚠️ Maximum risk concentrated in the near-wellbore zone. Priority protection: wellbore chemical treatment."))
        if R_at_Re > R_at_rw * 0.5 and R_at_rw < R_at_Re:
            recs.append(("warning", "📍 Higher risk at reservoir periphery. Consider inhibitor injection via the injection system."))
        Pe = st.session_state.P_e
        Pw = st.session_state.P_w
        depression = Pe - Pw
        if depression > 5.0:
            recs.append(("warning", f"🔩 Drawdown ΔP = {depression:.1f} MPa is high. Reducing drawdown will decrease supersaturation and deposition rate."))
        pH_val = st.session_state.pH
        if pH_val < 7.0:
            recs.append(("warning", "⚗️ pH < 7 — acidic environment. pH management can reduce CO₃²⁻ and scale risk."))
        if pH_val > 8.0:
            recs.append(("danger",  "⚗️ pH > 8 — alkaline environment increases carbonate scale risk. Consider water acidification."))
        if M_t > 1.0:
            recs.append(("danger",  f"💾 Expected scale mass over period: {M_t:.2f} t. Plan wellbore chemical treatment."))
        T_val = st.session_state.T
        if T_val > 80:
            recs.append(("warning", "🌡️ High temperature accelerates crystallisation. Tubing insulation may reduce risk."))

    return recs, severity

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_title, col_lang = st.columns([5, 1])
with col_title:
    st.markdown(f'<div class="hero-title">{L["app_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{L["app_subtitle"]}</div>', unsafe_allow_html=True)
with col_lang:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    if st.button(L["lang_toggle"], key="lang_btn"):
        st.session_state.lang = "en" if st.session_state.lang == "ru" else "ru"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STEP BAR
# ─────────────────────────────────────────────────────────────────────────────
step = st.session_state.step
steps = [L["step1"], L["step2"], L["step3"]]
step_html = '<div class="step-bar">'
for i, s in enumerate(steps):
    cls = "active" if i == step else ("done" if i < step else "")
    prefix = "✓ " if i < step else f"{i+1}. "
    step_html += f'<div class="step-item {cls}">{prefix}{s}</div>'
step_html += "</div>"
st.markdown(step_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT STEPS
# ─────────────────────────────────────────────────────────────────────────────

def _num(label, key, min_v, max_v, step_v=None, help_txt=None, fmt="%.4g"):
    val = st.number_input(
        label, value=float(st.session_state[key]),
        min_value=float(min_v), max_value=float(max_v),
        step=step_v, help=help_txt, key=f"inp_{key}", format=fmt,
    )
    st.session_state[key] = val
    return val

errors = []

# ── STEP 0: Chemistry ────────────────────────────────────────────────────────
if step == 0:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown(f"<h3>🧪 {L['step1']}</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        T_val = _num(L["T_label"], "T", 20.0, 120.0, 1.0, L["T_help"])
        if not (20 <= T_val <= 120):
            st.markdown(f'<div class="val-error">{L["err_T"]}</div>', unsafe_allow_html=True)
            errors.append("T")

        pH_val = _num(L["pH_label"], "pH", 1.0, 14.0, 0.1, L["pH_help"])
        if not (1 <= pH_val <= 14):
            st.markdown(f'<div class="val-error">{L["err_pH"]}</div>', unsafe_allow_html=True)
            errors.append("pH")

        _num(L["CCa_label"], "C_Ca", 0.0, 1e6, 10.0)

    with c2:
        _num(L["CHCO3_label"], "C_HCO3", 0.0, 1e6, 10.0)
        _num(L["CNaK_label"], "C_NaK",  0.0, 1e6, 50.0)
        _num(L["CCl_label"],  "C_Cl",   0.0, 1e6, 50.0)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(L["step_next"], disabled=len(errors) > 0):
        st.session_state.step = 1
        st.rerun()

# ── STEP 1: Reservoir ────────────────────────────────────────────────────────
elif step == 1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown(f"<h3>🪨 {L['step2']}</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        m_val = _num(L["m_label"], "m", 0.01, 0.99, 0.01, L["m_help"])
        if not (0 < m_val < 1):
            st.markdown(f'<div class="val-error">{L["err_m"]}</div>', unsafe_allow_html=True)
            errors.append("m")

        rock_idx = st.selectbox(
            L["rock_label"],
            options=range(len(L["rock_options"])),
            format_func=lambda i: L["rock_options"][i],
            index=st.session_state.rock_idx,
            key="inp_rock",
        )
        st.session_state.rock_idx = rock_idx

    with c2:
        _num(L["k_label"], "k", 0.01, 10000.0, 1.0)

    st.markdown("</div>", unsafe_allow_html=True)

    col_b, col_n = st.columns([1, 4])
    with col_b:
        if st.button(L["step_back"]):
            st.session_state.step = 0; st.rerun()
    with col_n:
        if st.button(L["step_next"], disabled=len(errors) > 0):
            st.session_state.step = 2; st.rerun()

# ── STEP 2: Hydrodynamics ────────────────────────────────────────────────────
elif step == 2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown(f"<h3>⛽ {L['step3']}</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        _num(L["Qw_label"], "Q_w", 0.1,  5000.0, 10.0)
        _num(L["Re_label"], "R_e", 10.0, 5000.0, 10.0)
        _num(L["rw_label"], "r_w", 0.01,  1.0,   0.01)
        _num(L["h_label"],  "h",   0.5,  500.0,  0.5)

    with c2:
        Pe_val = _num(L["Pe_label"], "P_e", 0.1, 100.0, 0.5)
        Pw_val = _num(L["Pw_label"], "P_w", 0.1, 100.0, 0.5)

        if Pw_val >= Pe_val:
            st.markdown(f'<div class="val-error">{L["err_Pw"]}</div>', unsafe_allow_html=True)
            errors.append("Pw")

        depression = Pe_val - Pw_val
        st.markdown(f"**{L['depression_label']}:** `{depression:.2f} МПа`" if st.session_state.lang == "ru"
                    else f"**{L['depression_label']}:** `{depression:.2f} MPa`")

        rw_v = st.session_state.r_w
        Re_v = st.session_state.R_e
        if rw_v >= Re_v:
            st.markdown(f'<div class="val-error">{L["err_rw"]}</div>', unsafe_allow_html=True)
            errors.append("rw")

        _num(L["t_label"], "t", 1.0, 36500.0, 30.0)

    st.markdown("</div>", unsafe_allow_html=True)

    col_b, col_n = st.columns([1, 4])
    with col_b:
        if st.button(L["step_back"]):
            st.session_state.step = 1; st.rerun()
    with col_n:
        if st.button(L["step_calc"], type="primary", disabled=len(errors) > 0):
            with st.spinner("Calculating..." if st.session_state.lang == "en" else "Идёт расчёт..."):
                s = st.session_state
                rock_type = L["rock_values"][s.rock_idx]
                res = run_model(
                    T=s.T, pH=s.pH, C_Ca=s.C_Ca, C_HCO3=s.C_HCO3,
                    C_NaK=s.C_NaK, C_Cl=s.C_Cl,
                    m=s.m, rock_type=rock_type, k=s.k,
                    Q_w=s.Q_w, R_e=s.R_e, r_w=s.r_w, h=s.h,
                    P_e=s.P_e, P_w=s.P_w, t=s.t,
                    n_points=150,
                )
                st.session_state.results = res
                st.session_state.step = 3
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif step == 3:
    res    = st.session_state.results
    r_arr  = res["r_arr"]
    SI_arr = res["SI_arr"]
    R_arr  = res["R_arr"]
    M_t    = res["M_t"]

    SI_max   = float(SI_arr.max())
    SI_min   = float(SI_arr.min())
    R_max    = float(R_arr.max())
    R_at_rw  = float(R_arr[0])

    # Pressure & velocity profiles (recomputed for display)
    s = st.session_state
    rock_type = L["rock_values"][s.rock_idx]
    r_log     = r_arr

    ln_outer  = math.log(s.R_e / s.r_w)
    P_arr     = s.P_e - (s.P_e - s.P_w) / ln_outer * np.log(s.R_e / r_log)
    u_arr     = s.Q_w / (2 * math.pi * r_log * s.h * 86400)
    v_arr     = u_arr / s.m

    # ── Recalc button ──────────────────────────────────────────────────────
    if st.button("← " + (L["step_back"] if False else (
            "Изменить параметры" if st.session_state.lang == "ru" else "Edit parameters"))):
        st.session_state.step = 0; st.rerun()

    # ── KPI CARDS ─────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-head">{L["results_title"]}</div>', unsafe_allow_html=True)

    card_color, badge_cls = _si_color(SI_max)
    si_label = L["si_under"] if SI_max < 0 else (L["si_near"] if SI_max < 0.5 else L["si_over"])
    R_color  = "green" if R_max == 0 else ("yellow" if R_max < 1 else "red")
    M_color  = "green" if M_t < 0.01 else ("yellow" if M_t < 10 else "red")

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card {card_color}">
        <div class="kpi-label">{L["kpi_SI_max"]}</div>
        <div class="kpi-value" style="color:{'#6c9cf5' if card_color=='blue' else ('#fbbf24' if card_color=='yellow' else '#f87171')}">
          {SI_max:+.3f}
        </div>
        <span class="kpi-badge {badge_cls}">{si_label}</span>
      </div>
      <div class="kpi-card {R_color}">
        <div class="kpi-label">{L["kpi_R_max"]}</div>
        <div class="kpi-value" style="color:{'#34d399' if R_color=='green' else ('#fbbf24' if R_color=='yellow' else '#f87171')}">
          {R_max:.3e}
        </div>
        <div class="kpi-sub">{L["kpi_R_unit"]}</div>
      </div>
      <div class="kpi-card {M_color}">
        <div class="kpi-label">{L["kpi_M"]}</div>
        <div class="kpi-value" style="color:{'#34d399' if M_color=='green' else ('#fbbf24' if M_color=='yellow' else '#f87171')}">
          {M_t:.4g}
        </div>
        <div class="kpi-sub">{L["kpi_M_unit"]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CHARTS ────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-head">📈 {("Профили параметров" if st.session_state.lang == "ru" else "Parameter Profiles")}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        L["chart_SI"], L["chart_P"], L["chart_v"], L["chart_R"]
    ])

    with tab1:
        fig = go.Figure()
        # color gradient by SI value
        fig.add_trace(go.Scatter(
            x=r_arr, y=SI_arr, mode="lines",
            line=dict(color="#6c9cf5", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(108,156,245,0.08)",
            hovertemplate=f"{L['r_axis']}: %{{x:.1f}}<br>SI: %{{y:.4f}}<extra></extra>",
        ))
        fig.add_hline(y=0, line=dict(color="#ef4444", width=1.5, dash="dash"),
                      annotation_text="SI = 0", annotation_position="bottom right",
                      annotation_font_color="#ef4444")
        fig.update_layout(xaxis_title=L["r_axis"], yaxis_title="SI(r)",
                          title=L["chart_SI"], **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = _plotly_line(r_arr, P_arr, "P(r)", "#a78bfa",
                           L["r_axis"], L["P_axis"], L["chart_P"])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = _plotly_line(r_arr, v_arr, "v(r)", "#34d399",
                           L["r_axis"], L["v_axis"], L["chart_v"])
        fig.update_layout(yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r_arr, y=R_arr, mode="lines",
            line=dict(color="#f87171", width=2.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.08)",
            hovertemplate=f"{L['r_axis']}: %{{x:.1f}}<br>R: %{{y:.4e}}<extra></extra>",
        ))
        fig.update_layout(xaxis_title=L["r_axis"], yaxis_title=L["R_axis"],
                          title=L["chart_R"], **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── HEATMAP (lightweight: filled Scatterpolar rings) ──────────────────
    st.markdown(f'<div class="section-head">🗺️ {L["chart_heatmap"]}</div>', unsafe_allow_html=True)
    st.caption(L["heatmap_note"])

    # Downsample to ≤60 radial rings — plenty for a visual, zero memory issues
    MAX_RINGS = 60
    step_r    = max(1, len(r_arr) // MAX_RINGS)
    r_ds      = r_arr[::step_r]
    R_ds      = R_arr[::step_r]

    R_max_hm  = R_ds.max() if R_ds.max() > 0 else 1.0

    # Colour ramp: dark-blue → green → orange → red
    def _ring_color(val, vmax):
        t = min(val / vmax, 1.0)
        if t < 0.33:
            r2, g2, b2 = int(30 + t/0.33*( 26-30)), int(58 + t/0.33*(107-58)), int(95 + t/0.33*(74-95))
        elif t < 0.66:
            tt = (t-0.33)/0.33
            r2, g2, b2 = int(26 + tt*(180-26)), int(107 + tt*(83-107)), int(74 + tt*(9-74))
        else:
            tt = (t-0.66)/0.34
            r2, g2, b2 = int(180 + tt*(239-180)), int(83 + tt*(68-83)), int(9 + tt*(68-9))
        return f"rgba({r2},{g2},{b2},0.85)"

    theta_ring = list(np.linspace(0, 360, 72, endpoint=False)) + [0]  # close the ring

    fig_hm = go.Figure()

    # Draw rings from outside in so inner rings paint over outer ones
    for i in range(len(r_ds)-1, -1, -1):
        color = _ring_color(R_ds[i], R_max_hm)
        fig_hm.add_trace(go.Scatterpolar(
            r=[r_ds[i]] * len(theta_ring),
            theta=theta_ring,
            mode="lines",
            fill="toself" if i == 0 else "tonextr",
            fillcolor=color,
            line=dict(color=color, width=0),
            hovertemplate=f"r = {r_ds[i]:.1f} m<br>R = {R_ds[i]:.3e}<extra></extra>",
            showlegend=False,
        ))

    # Wellbore marker at centre
    fig_hm.add_trace(go.Scatterpolar(
        r=[0], theta=[0], mode="markers+text",
        marker=dict(size=12, color="white", symbol="circle"),
        text=["⛳"], textposition="middle center",
        hoverinfo="skip", showlegend=False,
    ))

    # Invisible colour axis for the colour bar using a tiny scatter
    fig_hm.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale=[
                [0.0,  "#1e3a5f"],
                [0.33, "#1a6b4a"],
                [0.66, "#b45309"],
                [1.0,  "#ef4444"],
            ],
            cmin=0, cmax=R_max_hm,
            colorbar=dict(
                title=dict(text=L["R_axis"], font=dict(color="#9ca3af", size=11)),
                tickfont=dict(color="#9ca3af", family="JetBrains Mono", size=10),
                bgcolor="rgba(0,0,0,0)",
                thickness=14,
                len=0.75,
            ),
            showscale=True,
            color=[0],
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_hm.update_layout(
        polar=dict(
            bgcolor="#0d1017",
            radialaxis=dict(
                visible=True, showgrid=True, gridcolor="#1e2230",
                tickfont=dict(color="#6b7280", size=10, family="JetBrains Mono"),
                ticksuffix=" m",
            ),
            angularaxis=dict(showgrid=False, showticklabels=False),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#9ca3af"),
        height=480,
        margin=dict(l=40, r=80, t=40, b=40),
        hoverlabel=dict(bgcolor="#1e2230", bordercolor="#2d3748",
                        font=dict(family="JetBrains Mono", color="#e8eaf0")),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── SENSITIVITY ANALYSIS ──────────────────────────────────────────────
    st.markdown(f'<div class="section-head">🔬 {L["sens_title"]}</div>', unsafe_allow_html=True)
    st.caption(L["sens_note"])

    if st.button(L["sens_btn"]):
        with st.spinner("Running sensitivity..." if st.session_state.lang == "en" else "Анализ чувствительности..."):
            base = dict(T=s.T, pH=s.pH, C_Ca=s.C_Ca, C_HCO3=s.C_HCO3,
                        C_NaK=s.C_NaK, C_Cl=s.C_Cl, m=s.m,
                        rock_type=L["rock_values"][s.rock_idx], k=s.k,
                        Q_w=s.Q_w, R_e=s.R_e, r_w=s.r_w, h=s.h,
                        P_e=s.P_e, P_w=s.P_w, t=s.t, n_points=100)

            def _sweep(param, values, **override):
                R_vals = []
                for v in values:
                    kw = {**base, **override, param: v}
                    try:
                        r = run_model(**kw)
                        R_vals.append(float(r["R_arr"][0]))
                    except Exception:
                        R_vals.append(0.0)
                return R_vals

            pH_range  = np.linspace(5.5, 9.5, 40)
            T_range   = np.linspace(20, 120, 40)
            Pw_range  = np.linspace(s.P_w * 0.5, s.P_e * 0.95, 40)

            R_pH  = _sweep("pH",  pH_range)
            R_T   = _sweep("T",   T_range)
            R_Pw  = _sweep("P_w", Pw_range)

        fig_s = make_subplots(rows=1, cols=3,
            subplot_titles=[L["sens_pH"], L["sens_T"], L["sens_Pw"]])

        def _add_sens(fig, col, x, y, color, xlabel):
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color=color, width=2.5),
                fill="tozeroy", fillcolor="rgba(108,156,245,0.1)",
                hovertemplate=f"{xlabel}: %{{x:.2f}}<br>R: %{{y:.3e}}<extra></extra>",
            ), row=1, col=col)

        _add_sens(fig_s, 1, pH_range,  R_pH,  "#6c9cf5",  "pH")
        _add_sens(fig_s, 2, T_range,   R_T,   "#a78bfa",  "T °C")
        _add_sens(fig_s, 3, Pw_range,  R_Pw,  "#34d399",  "Pw (MPa)")

        fig_s.update_layout(
            height=360, showlegend=False,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
        )
        for ax in ["xaxis","yaxis","xaxis2","yaxis2","xaxis3","yaxis3"]:
            fig_s.update_layout(**{ax: dict(gridcolor="#1e2230", linecolor="#1e2230",
                                            zerolinecolor="#2d3748",
                                            color="#9ca3af")})
        st.plotly_chart(fig_s, use_container_width=True)

    # ── ACCUMULATION ANIMATION ────────────────────────────────────────────
    st.markdown(f'<div class="section-head">⏱ {L["anim_title"]}</div>', unsafe_allow_html=True)

    years = st.slider(L["anim_years"], 1, 20, 5, key="anim_years_slider")
    if st.button(L["anim_btn"]):
        t_points = np.linspace(0, years * 365, 60)
        M_points = [float(_calc_total_mass_fast(r_arr, R_arr, s.h, t_i))
                    for t_i in t_points]

        fig_anim = go.Figure()
        fig_anim.add_trace(go.Scatter(
            x=t_points / 365, y=M_points, mode="lines",
            line=dict(color="#f59e0b", width=3),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
        ))
        fig_anim.update_layout(
            xaxis_title="t (лет)" if st.session_state.lang == "ru" else "t (years)",
            yaxis_title="M(t) (т)" if st.session_state.lang == "ru" else "M(t) (tonnes)",
            title="M(t)" ,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_anim, use_container_width=True)

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────
    st.markdown(f'<div class="section-head">💡 {L["rec_title"]}</div>', unsafe_allow_html=True)

    recs, sev = _get_recommendations(r_arr, SI_arr, R_arr, M_t, res, st.session_state.lang)
    if not recs:
        st.markdown(f'<div class="rec-card ok">{L["rec_none"]}</div>', unsafe_allow_html=True)
    else:
        for cls, text in recs:
            st.markdown(f'<div class="rec-card {cls}">{text}</div>', unsafe_allow_html=True)

    # ── DETAILS EXPANDER ──────────────────────────────────────────────────
    with st.expander(L["details_expander"]):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("SI_min", f"{SI_min:+.4f}")
            st.metric("SI_max", f"{SI_max:+.4f}")
            st.metric("R at r_w", f"{R_at_rw:.4e}")
        with col_d2:
            st.metric("R_max",  f"{R_max:.4e}")
            st.metric("M(t)",   f"{M_t:.4e} т")
            st.metric("n_r",    "300")
            st.metric("M(t)",   f"{M_t:.4e} т")
            st.metric("n_r",    "300")


