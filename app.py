"""
CaCO₃ Scale Deposition Predictor  —  v2.2
Streamlit UI. Совместим с model.py v2.2.
"""

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import run_model

# ═══════════════════════════════════════════════════════════════════════════════
# ЛОКАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════
LANG = {
    "ru": {
        "app_title": "CaCO₃ Scale Predictor",
        "app_sub"  : "Прогнозирование риска карбонатного солеотложения в поровом пространстве коллектора",
        "lang_btn" : "🇬🇧 English",
        "step1": "Химия воды", "step2": "ФЕС", "step3": "Гидродинамика",
        "btn_next": "Далее →", "btn_back": "← Назад",
        "btn_calc": "▶  Рассчитать", "btn_edit": "← Изменить параметры",
        # Химия
        "T_lbl"    : "Температура пласта (°C)",      "T_help"   : "20–120 °C",
        "pH_lbl"   : "pH пластовой воды",             "pH_help"  : "1–14",
        "CCa_lbl"  : "Концентрация Ca²⁺ (мг/л)",
        "CHCO3_lbl": "Концентрация HCO₃⁻ (мг/л)",
        "CNaK_lbl" : "Концентрация Na⁺ + K⁺ (мг/л)",
        "CCl_lbl"  : "Концентрация Cl⁻ (мг/л)",
        # ФЕС
        "m_lbl"   : "Пористость (доли)",  "m_help": "0–1",
        "rock_lbl": "Тип коллектора",
        "rock_opt": ["Песчаник", "Карбонат"],
        "rock_val": ["sandstone", "carbonate"],
        "k_lbl"   : "Проницаемость по воде (мД)",
        # Гидро
        "Qw_lbl" : "Дебит воды (м³/сут)",
        "Re_lbl" : "Радиус контура питания Rₑ (м)",
        "rw_lbl" : "Радиус скважины rw (м)",
        "h_lbl"  : "Толщина пласта h (м)",
        "Pe_lbl" : "Пластовое давление Pₑ (МПа)",
        "Pw_lbl" : "Забойное давление Pw (МПа)",
        "t_lbl"  : "Период эксплуатации t (сут)",
        "dP_lbl" : "Депрессия ΔP = Pₑ − Pw",
        "r_pbz_lbl": "Радиус призабойной зоны (м)",
        # Ошибки валидации
        "err_T"  : "⛔ Температура: допустимо 20–120 °C",
        "err_pH" : "⛔ pH: допустимо 1–14",
        "err_m"  : "⛔ Пористость: допустимо 0–1 (исключая границы)",
        "err_Pw" : "⛔ Забойное давление должно быть меньше пластового",
        "err_rw" : "⛔ Радиус скважины должен быть меньше радиуса контура",
        # KPI
        "res_title"  : "Результаты расчёта",
        "kpi_SI"     : "Максимальный SI",
        "kpi_R"      : "Макс. скорость R(r)",
        "kpi_Mpbz"   : "Масса осадка в ПЗП",
        "kpi_R_unit" : "кг/(м³·сут)",
        "kpi_M_unit" : "кг",
        "si_neg" : "Недонасыщено",
        "si_low" : "Слабый риск",
        "si_med" : "Умеренный риск",
        "si_high": "Высокий риск!",
        # Графики
        "tab_SI" : "SI(r)", "tab_P": "P(r)", "tab_v": "v(r)", "tab_R": "R(r)",
        "chart_SI": "Индекс насыщения SI(r)",
        "chart_P" : "Давление P(r)",
        "chart_v" : "Поровая скорость v(r)",
        "chart_R" : "Скорость солеотложения R(r)",
        "r_ax": "Радиус r (м)", "SI_ax": "SI(r)", "P_ax": "P (МПа)",
        "v_ax": "v (м/с)",     "R_ax" : "R (кг/(м³·сут))",
        "hmap_title": "Карта зоны риска (радиальный разрез)",
        "hmap_note" : "Центр — скважина. Цвет по R(r). Синее кольцо — граница ПЗП.",
        # Чувствительность
        "sens_title": "Анализ чувствительности",
        "sens_note" : "R у забоя (r = rw) при варьировании одного параметра. Остальные — как введено.",
        "sens_btn"  : "Запустить анализ",
        "sens_pH"   : "R(rw) vs pH",
        "sens_T"    : "R(rw) vs T (°C)",
        "sens_Pw"   : "R(rw) vs Pw (МПа)",
        # Накопление
        "acc_title" : "Накопление осадка во времени",
        "acc_btn"   : "Построить M(t)",
        "acc_years" : "Горизонт (лет)",
        "acc_pbz"   : "Масса в ПЗП (кг)",
        "acc_all"   : "Масса весь пласт (кг)",
        # Рекомендации
        "rec_title": "Инженерные рекомендации",
        "rec_ok"   : "✅ Риск солеотложения отсутствует. Текущие условия благоприятны.",
        # Детали
        "det_title": "Детали расчёта",
        "det_Sv"   : "Удельная поверхность Sᵥ",
        "det_SIe"  : "Базовый SI_e",
        "det_SIrw" : "SI(rw) — у забоя",
        "det_Rrw"  : "R(rw) — у забоя",
        "det_Rmax" : "R_max — максимум",
        "det_Mt"   : "M(t) весь пласт",
        "det_eta"  : "η (доля осаждения)",
        "det_Mpbz" : "M в ПЗП",
    },
    "en": {
        "app_title": "CaCO₃ Scale Predictor",
        "app_sub"  : "Carbonate scale deposition risk prediction in reservoir pore space",
        "lang_btn" : "🇷🇺 Русский",
        "step1": "Water Chemistry", "step2": "Reservoir", "step3": "Hydrodynamics",
        "btn_next": "Next →", "btn_back": "← Back",
        "btn_calc": "▶  Calculate", "btn_edit": "← Edit parameters",
        "T_lbl"    : "Reservoir Temperature (°C)", "T_help"   : "20–120 °C",
        "pH_lbl"   : "Formation Water pH",         "pH_help"  : "1–14",
        "CCa_lbl"  : "Ca²⁺ Concentration (mg/L)",
        "CHCO3_lbl": "HCO₃⁻ Concentration (mg/L)",
        "CNaK_lbl" : "Na⁺ + K⁺ Concentration (mg/L)",
        "CCl_lbl"  : "Cl⁻ Concentration (mg/L)",
        "m_lbl"   : "Porosity (fraction)", "m_help": "0–1",
        "rock_lbl": "Reservoir Type",
        "rock_opt": ["Sandstone", "Carbonate"],
        "rock_val": ["sandstone", "carbonate"],
        "k_lbl"   : "Water Permeability (mD)",
        "Qw_lbl" : "Water Rate (m³/day)",
        "Re_lbl" : "Drainage Radius Rₑ (m)",
        "rw_lbl" : "Wellbore Radius rw (m)",
        "h_lbl"  : "Net Pay Thickness h (m)",
        "Pe_lbl" : "Reservoir Pressure Pₑ (MPa)",
        "Pw_lbl" : "BHP Pw (MPa)",
        "t_lbl"  : "Production Period t (days)",
        "dP_lbl" : "Drawdown ΔP = Pₑ − Pw",
        "r_pbz_lbl": "Near-wellbore Zone Radius (m)",
        "err_T"  : "⛔ Temperature must be 20–120 °C",
        "err_pH" : "⛔ pH must be 1–14",
        "err_m"  : "⛔ Porosity must be between 0 and 1 (exclusive)",
        "err_Pw" : "⛔ BHP must be less than reservoir pressure",
        "err_rw" : "⛔ Wellbore radius must be less than drainage radius",
        "res_title"  : "Calculation Results",
        "kpi_SI"     : "Maximum SI",
        "kpi_R"      : "Max Rate R(r)",
        "kpi_Mpbz"   : "Near-wellbore Scale Mass",
        "kpi_R_unit" : "kg/(m³·day)",
        "kpi_M_unit" : "kg",
        "si_neg" : "Undersaturated",
        "si_low" : "Low risk",
        "si_med" : "Moderate risk",
        "si_high": "High risk!",
        "tab_SI": "SI(r)", "tab_P": "P(r)", "tab_v": "v(r)", "tab_R": "R(r)",
        "chart_SI": "Saturation Index SI(r)",
        "chart_P" : "Pressure Profile P(r)",
        "chart_v" : "Pore Velocity v(r)",
        "chart_R" : "Deposition Rate R(r)",
        "r_ax": "Radius r (m)", "SI_ax": "SI(r)", "P_ax": "P (MPa)",
        "v_ax": "v (m/s)",     "R_ax" : "R (kg/(m³·day))",
        "hmap_title": "Risk Zone Map (Radial Cross-Section)",
        "hmap_note" : "Center — wellbore. Color by R(r). Blue ring — near-wellbore boundary.",
        "sens_title": "Sensitivity Analysis",
        "sens_note" : "R at wellbore (r = rw) varying one parameter at a time.",
        "sens_btn"  : "Run Analysis",
        "sens_pH"   : "R(rw) vs pH",
        "sens_T"    : "R(rw) vs T (°C)",
        "sens_Pw"   : "R(rw) vs Pw (MPa)",
        "acc_title" : "Scale Accumulation Over Time",
        "acc_btn"   : "Build M(t) curve",
        "acc_years" : "Horizon (years)",
        "acc_pbz"   : "Near-wellbore mass (kg)",
        "acc_all"   : "Total reservoir mass (kg)",
        "rec_title": "Engineering Recommendations",
        "rec_ok"   : "✅ No scale risk detected. Current conditions are favourable.",
        "det_title": "Calculation Details",
        "det_Sv"   : "Specific Surface Sᵥ",
        "det_SIe"  : "Base SI_e",
        "det_SIrw" : "SI(rw) — at wellbore",
        "det_Rrw"  : "R(rw) — at wellbore",
        "det_Rmax" : "R_max — profile max",
        "det_Mt"   : "M(t) whole reservoir",
        "det_eta"  : "η (deposition fraction)",
        "det_Mpbz" : "M near-wellbore zone",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CaCO₃ Scale Predictor",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#0d0f14;color:#e8eaf0;}
.hero-title{font-family:'DM Serif Display',serif;font-size:clamp(1.8rem,4vw,3rem);
  letter-spacing:-.02em;background:linear-gradient(135deg,#e8eaf0 30%,#6c9cf5 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  line-height:1.1;margin-bottom:.2rem;}
.hero-sub{font-size:.9rem;color:#6b7280;font-weight:300;margin-bottom:2rem;}
.step-bar{display:flex;margin-bottom:1.5rem;border-radius:10px;overflow:hidden;border:1px solid #1e2230;}
.step-item{flex:1;padding:.65rem .5rem;text-align:center;font-size:.72rem;font-weight:600;
  color:#4b5563;background:#131620;letter-spacing:.05em;text-transform:uppercase;
  border-right:1px solid #1e2230;}
.step-item:last-child{border-right:none;}
.step-item.active{background:#1a2640;color:#6c9cf5;border-bottom:2px solid #6c9cf5;}
.step-item.done{background:#131f2e;color:#34d399;}
.input-card{background:#131620;border:1px solid #1e2230;border-radius:14px;
  padding:1.5rem 1.75rem;margin-bottom:1rem;}
.input-card h3{font-family:'DM Serif Display',serif;font-size:1.1rem;color:#c9d1e0;
  margin-bottom:1rem;padding-bottom:.4rem;border-bottom:1px solid #1e2230;}
.val-err{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.35);
  color:#fca5a5;border-radius:7px;padding:.45rem .8rem;font-size:.82rem;margin-top:.2rem;}
.kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem;margin-bottom:1.8rem;}
.kpi-card{background:#131620;border:1px solid #1e2230;border-radius:14px;
  padding:1.3rem 1.4rem;position:relative;overflow:hidden;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.kc-blue::before{background:#3b82f6;} .kc-green::before{background:#10b981;}
.kc-yellow::before{background:#f59e0b;} .kc-red::before{background:#ef4444;}
.kpi-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;
  color:#6b7280;margin-bottom:.4rem;}
.kpi-val{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:600;
  line-height:1;margin-bottom:.35rem;}
.kpi-sub{font-size:.78rem;color:#6b7280;}
.kpi-badge{display:inline-block;padding:.18rem .55rem;border-radius:999px;
  font-size:.68rem;font-weight:600;letter-spacing:.03em;margin-top:.35rem;}
.bd-blue{background:rgba(59,130,246,.15);color:#6c9cf5;}
.bd-green{background:rgba(16,185,129,.15);color:#34d399;}
.bd-yellow{background:rgba(245,158,11,.15);color:#fbbf24;}
.bd-red{background:rgba(239,68,68,.15);color:#f87171;}
.sec-head{font-family:'DM Serif Display',serif;font-size:1.2rem;color:#c9d1e0;
  margin:1.8rem 0 .9rem;padding-bottom:.35rem;border-bottom:1px solid #1e2230;}
.rec-card{background:#131620;border:1px solid #1e2230;border-left:3px solid #6c9cf5;
  border-radius:9px;padding:.85rem 1.1rem;margin-bottom:.5rem;
  font-size:.88rem;color:#c9d1e0;line-height:1.5;}
.rec-card.warn{border-left-color:#f59e0b;} .rec-card.danger{border-left-color:#ef4444;}
.rec-card.ok{border-left-color:#10b981;}
div.stButton>button{background:#1a2640;color:#6c9cf5;border:1px solid #2d4a7a;
  border-radius:9px;padding:.55rem 1.3rem;font-family:'DM Sans',sans-serif;
  font-weight:600;font-size:.88rem;transition:all .18s;}
div.stButton>button:hover{background:#233457;border-color:#6c9cf5;color:#93b8ff;}
div.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1a3a6e,#1e4a94);
  color:#93b8ff;border:1px solid #3b6ab8;font-size:.95rem;padding:.7rem 1.8rem;}
[data-testid="stExpander"]{background:#131620;border:1px solid #1e2230;border-radius:11px;}
.stTabs [data-baseweb="tab-list"]{background:#131620;border-radius:9px;}
.stTabs [data-baseweb="tab"]{color:#6b7280;}
.stTabs [aria-selected="true"]{color:#6c9cf5 !important;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
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

L     = LANG[st.session_state.lang]
IS_RU = st.session_state.lang == "ru"

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY БАЗОВЫЙ СТИЛЬ
# ═══════════════════════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1017",
    font=dict(family="DM Sans,sans-serif", color="#9ca3af", size=12),
    xaxis=dict(gridcolor="#1e2230", zerolinecolor="#2d3748", linecolor="#1e2230"),
    yaxis=dict(gridcolor="#1e2230", zerolinecolor="#2d3748", linecolor="#1e2230"),
    margin=dict(l=50, r=20, t=40, b=50),
    hoverlabel=dict(bgcolor="#1e2230", bordercolor="#2d3748",
                    font=dict(family="JetBrains Mono", color="#e8eaf0")),
)

# ═══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════
def _si_style(si_max):
    if si_max < 0:    return "kc-blue",   "bd-blue",   L["si_neg"],  "#6c9cf5"
    if si_max < 0.5:  return "kc-green",  "bd-green",  L["si_low"],  "#34d399"
    if si_max < 1.5:  return "kc-yellow", "bd-yellow", L["si_med"],  "#fbbf24"
    return                   "kc-red",    "bd-red",    L["si_high"], "#f87171"

def _risk_style(R_max):
    if R_max < 1e-4: return "kc-green",  "#34d399"
    if R_max < 1e-2: return "kc-yellow", "#fbbf24"
    return                  "kc-red",    "#f87171"

def _mass_style(M):
    if M < 1.0:    return "kc-green",  "#34d399"
    if M < 100.0:  return "kc-yellow", "#fbbf24"
    return                "kc-red",    "#f87171"

def _calc_M_pbz(r_arr, R_arr, h, t, r_pbz, r_w):
    """Масса осадка в призабойной зоне r < r_pbz [кг]."""
    mask = r_arr <= max(r_pbz, r_w * 2)
    if mask.sum() < 2:
        # Если точек мало — оцениваем по первой точке
        return float(R_arr[0]) * math.pi * (r_pbz**2 - r_w**2) * h * t
    integrand = R_arr[mask] * 2.0 * math.pi * r_arr[mask] * h
    return t * float(np.trapezoid(integrand, r_arr[mask]))

def _num(label, key, lo, hi, step=None, help_txt=None, fmt="%.4g"):
    val = st.number_input(
        label, value=float(st.session_state[key]),
        min_value=float(lo), max_value=float(hi),
        step=step, help=help_txt, key=f"ni_{key}", format=fmt,
    )
    st.session_state[key] = val
    return val

def _call_model(n_points=150):
    """Вызов run_model с текущими параметрами сессии."""
    s = st.session_state
    return run_model(
        T=s.T, pH=s.pH, C_Ca=s.C_Ca, C_HCO3=s.C_HCO3,
        C_NaK=s.C_NaK, C_Cl=s.C_Cl,
        m=s.m, rock_type=L["rock_val"][s.rock_idx], k=s.k,
        Q_w=s.Q_w, R_e=s.R_e, r_w=s.r_w, h=s.h,
        P_e=s.P_e, P_w=s.P_w, t=s.t,
        n_points=n_points,
    )

def _recommendations(SI_arr, R_arr, M_pbz):
    recs = []
    SI_max = float(SI_arr.max())
    R_rw   = float(R_arr[0])
    R_max  = float(R_arr.max())
    s      = st.session_state
    dP     = s.P_e - s.P_w
    lang   = st.session_state.lang

    if SI_max <= 0:
        return [], "ok"

    if lang == "ru":
        if SI_max > 0.5:
            recs.append(("warn",
                "📊 SI > 0.5 — раствор пересыщен. Необходим регулярный мониторинг химсостава воды."))
        if SI_max > 1.5:
            recs.append(("danger",
                "🧪 SI > 1.5 — высокий риск нуклеации. Рекомендуется ингибитор (фосфонаты, полиакрилаты)."))
        if R_rw >= R_max * 0.6:
            recs.append(("danger",
                "⚠️ Максимальный риск сосредоточен в призабойной зоне. Приоритет: обработка скважины ингибитором."))
        if dP > 5.0:
            recs.append(("warn",
                f"🔩 Депрессия ΔP = {dP:.1f} МПа высокая. Снижение депрессии уменьшит пересыщение."))
        if s.pH > 8.0:
            recs.append(("danger",
                "⚗️ pH > 8 — щелочная среда усиливает риск. Рассмотрите подкисление воды."))
        if s.pH < 6.5:
            recs.append(("warn",
                "⚗️ pH < 6.5 — кислая среда. Контроль pH снижает концентрацию CO₃²⁻."))
        if s.T > 80:
            recs.append(("warn",
                "🌡️ Высокая температура ускоряет кристаллизацию. Рассмотрите теплоизоляцию НКТ."))
        if M_pbz > 50:
            recs.append(("danger",
                f"💾 Масса осадка в ПЗП: {M_pbz:.1f} кг. Планируйте химическую обработку скважины."))
        if L["rock_val"][s.rock_idx] == "carbonate":
            recs.append(("warn",
                "🪨 Карбонатный коллектор: химическое сродство с CaCO₃ снижает барьер нуклеации."))
    else:
        if SI_max > 0.5:
            recs.append(("warn",
                "📊 SI > 0.5 — supersaturated. Regular water chemistry monitoring recommended."))
        if SI_max > 1.5:
            recs.append(("danger",
                "🧪 SI > 1.5 — high nucleation risk. Consider scale inhibitor (phosphonates, polyacrylates)."))
        if R_rw >= R_max * 0.6:
            recs.append(("danger",
                "⚠️ Max risk concentrated near wellbore. Priority: wellbore chemical squeeze treatment."))
        if dP > 5.0:
            recs.append(("warn",
                f"🔩 Drawdown ΔP = {dP:.1f} MPa is high. Reducing drawdown will decrease supersaturation."))
        if s.pH > 8.0:
            recs.append(("danger",
                "⚗️ pH > 8 — alkaline environment increases risk. Consider injection water acidification."))
        if s.pH < 6.5:
            recs.append(("warn",
                "⚗️ pH < 6.5 — acidic environment. pH management reduces CO₃²⁻ concentration."))
        if s.T > 80:
            recs.append(("warn",
                "🌡️ High temperature accelerates crystallisation. Consider tubing insulation."))
        if M_pbz > 50:
            recs.append(("danger",
                f"💾 Near-wellbore scale mass: {M_pbz:.1f} kg. Plan wellbore chemical treatment."))
        if L["rock_val"][s.rock_idx] == "carbonate":
            recs.append(("warn",
                "🪨 Carbonate reservoir: chemical affinity with CaCO₃ lowers nucleation barrier."))
    return recs, ("danger" if SI_max > 1.5 else "warn")

# ═══════════════════════════════════════════════════════════════════════════════
# ЗАГОЛОВОК
# ═══════════════════════════════════════════════════════════════════════════════
col_h, col_l = st.columns([5, 1])
with col_h:
    st.markdown(f'<div class="hero-title">{L["app_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{L["app_sub"]}</div>',    unsafe_allow_html=True)
with col_l:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button(L["lang_btn"], key="lang_toggle"):
        st.session_state.lang = "en" if IS_RU else "ru"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ПОЛОСА ШАГОВ
# ═══════════════════════════════════════════════════════════════════════════════
step  = st.session_state.step
steps = [L["step1"], L["step2"], L["step3"]]
bar   = '<div class="step-bar">'
for i, sname in enumerate(steps):
    cls = "active" if i == step else ("done" if i < step else "")
    pfx = "✓ " if i < step else f"{i+1}. "
    bar += f'<div class="step-item {cls}">{pfx}{sname}</div>'
bar += "</div>"
st.markdown(bar, unsafe_allow_html=True)

errors = []

# ═══════════════════════════════════════════════════════════════════════════════
# ШАГ 0 — ХИМИЯ ВОДЫ
# ═══════════════════════════════════════════════════════════════════════════════
if step == 0:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown(f"<h3>🧪 {L['step1']}</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        T_v = _num(L["T_lbl"],  "T",   20.0, 120.0, 1.0,  L["T_help"])
        if not (20 <= T_v <= 120):
            st.markdown(f'<div class="val-err">{L["err_T"]}</div>', unsafe_allow_html=True)
            errors.append("T")
        pH_v = _num(L["pH_lbl"], "pH",  1.0,  14.0,  0.1,  L["pH_help"])
        if not (1 <= pH_v <= 14):
            st.markdown(f'<div class="val-err">{L["err_pH"]}</div>', unsafe_allow_html=True)
            errors.append("pH")
        _num(L["CCa_lbl"],   "C_Ca",   0.0, 1e6, 10.0)
    with c2:
        _num(L["CHCO3_lbl"], "C_HCO3", 0.0, 1e6, 10.0)
        _num(L["CNaK_lbl"],  "C_NaK",  0.0, 1e6, 50.0)
        _num(L["CCl_lbl"],   "C_Cl",   0.0, 1e6, 50.0)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button(L["btn_next"], disabled=bool(errors)):
        st.session_state.step = 1; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ШАГ 1 — ФЕС
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown(f"<h3>🪨 {L['step2']}</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        m_v = _num(L["m_lbl"], "m", 0.01, 0.99, 0.01, L["m_help"])
        if not (0 < m_v < 1):
            st.markdown(f'<div class="val-err">{L["err_m"]}</div>', unsafe_allow_html=True)
            errors.append("m")
        rock_idx = st.selectbox(
            L["rock_lbl"],
            options=range(len(L["rock_opt"])),
            format_func=lambda i: L["rock_opt"][i],
            index=st.session_state.rock_idx,
            key="sel_rock",
        )
        st.session_state.rock_idx = rock_idx
    with c2:
        _num(L["k_lbl"], "k", 0.01, 10000.0, 1.0)
    st.markdown("</div>", unsafe_allow_html=True)
    cb, cn = st.columns([1, 4])
    with cb:
        if st.button(L["btn_back"]): st.session_state.step = 0; st.rerun()
    with cn:
        if st.button(L["btn_next"], disabled=bool(errors)): st.session_state.step = 2; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ШАГ 2 — ГИДРОДИНАМИКА
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown(f"<h3>⛽ {L['step3']}</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        _num(L["Qw_lbl"], "Q_w", 0.1,  5000.0, 10.0)
        _num(L["Re_lbl"], "R_e", 10.0, 5000.0, 10.0)
        _num(L["rw_lbl"], "r_w", 0.01, 1.0,    0.01)
        _num(L["h_lbl"],  "h",   0.5,  500.0,  0.5)
    with c2:
        Pe_v = _num(L["Pe_lbl"], "P_e", 0.1, 100.0, 0.5)
        Pw_v = _num(L["Pw_lbl"], "P_w", 0.1, 100.0, 0.5)
        if Pw_v >= Pe_v:
            st.markdown(f'<div class="val-err">{L["err_Pw"]}</div>', unsafe_allow_html=True)
            errors.append("Pw")
        unit_p = "МПа" if IS_RU else "MPa"
        st.markdown(f"**{L['dP_lbl']}:** `{Pe_v - Pw_v:.2f} {unit_p}`")
        if st.session_state.r_w >= st.session_state.R_e:
            st.markdown(f'<div class="val-err">{L["err_rw"]}</div>', unsafe_allow_html=True)
            errors.append("rw")
        _num(L["t_lbl"],     "t",     1.0,  36500.0, 30.0)
        _num(L["r_pbz_lbl"], "r_pbz", 1.0,  200.0,   1.0)
    st.markdown("</div>", unsafe_allow_html=True)
    cb, cn = st.columns([1, 4])
    with cb:
        if st.button(L["btn_back"]): st.session_state.step = 1; st.rerun()
    with cn:
        if st.button(L["btn_calc"], type="primary", disabled=bool(errors)):
            with st.spinner("Расчёт..." if IS_RU else "Calculating..."):
                st.session_state.results = _call_model()
                st.session_state.step = 3
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ШАГ 3 — РЕЗУЛЬТАТЫ
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 3:
    res    = st.session_state.results
    r_arr  = res["r_arr"]
    SI_arr = res["SI_arr"]
    R_arr  = res["R_arr"]
    P_arr  = res["P_arr"]
    v_arr  = res["v_arr"]
    M_t    = res["M_t"]
    SI_max = float(SI_arr.max())
    R_max  = float(R_arr.max())
    s      = st.session_state

    # M в призабойной зоне — считаем здесь, не в модели
    M_pbz = _calc_M_pbz(r_arr, R_arr, s.h, s.t, s.r_pbz, s.r_w)

    if st.button(L["btn_edit"]):
        st.session_state.step = 0; st.rerun()

    # ── KPI ────────────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-head">{L["res_title"]}</div>', unsafe_allow_html=True)

    kc_si, bd_si, lbl_si, col_si = _si_style(SI_max)
    kc_r,  col_r                 = _risk_style(R_max)
    kc_m,  col_m                 = _mass_style(M_pbz)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card {kc_si}">
        <div class="kpi-label">{L["kpi_SI"]}</div>
        <div class="kpi-val" style="color:{col_si}">{SI_max:+.3f}</div>
        <span class="kpi-badge {bd_si}">{lbl_si}</span>
      </div>
      <div class="kpi-card {kc_r}">
        <div class="kpi-label">{L["kpi_R"]}</div>
        <div class="kpi-val" style="color:{col_r}">{R_max:.3e}</div>
        <div class="kpi-sub">{L["kpi_R_unit"]}</div>
      </div>
      <div class="kpi-card {kc_m}">
        <div class="kpi-label">{L["kpi_Mpbz"]} (r &lt; {s.r_pbz:.0f} м)</div>
        <div class="kpi-val" style="color:{col_m}">{M_pbz:.1f}</div>
        <div class="kpi-sub">{L["kpi_M_unit"]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ГРАФИКИ ────────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-head">📈 {"Профили параметров" if IS_RU else "Parameter Profiles"}</div>',
                unsafe_allow_html=True)

    tab_si, tab_p, tab_v, tab_r = st.tabs([L["tab_SI"], L["tab_P"], L["tab_v"], L["tab_R"]])

    with tab_si:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r_arr, y=SI_arr, mode="lines",
            line=dict(color="#6c9cf5", width=2.5),
            fill="tozeroy", fillcolor="rgba(108,156,245,0.07)",
            hovertemplate=f"{L['r_ax']}: %{{x:.1f}}<br>SI: %{{y:.4f}}<extra></extra>",
        ))
        fig.add_hline(y=0,   line=dict(color="#ef4444", width=1.5, dash="dash"),
                      annotation_text="SI=0",   annotation_font_color="#ef4444")
        fig.add_hline(y=0.5, line=dict(color="#f59e0b", width=1,   dash="dot"),
                      annotation_text="SI=0.5", annotation_font_color="#f59e0b")
        fig.update_layout(xaxis_title=L["r_ax"], yaxis_title=L["SI_ax"],
                          title=L["chart_SI"], **PL)
        st.plotly_chart(fig, use_container_width=True)

    with tab_p:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r_arr, y=P_arr, mode="lines", name="P(r)",
            line=dict(color="#a78bfa", width=2.5),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.07)",
            hovertemplate=f"{L['r_ax']}: %{{x:.1f}}<br>{L['P_ax']}: %{{y:.3f}}<extra></extra>",
        ))
        fig.update_layout(xaxis_title=L["r_ax"], yaxis_title=L["P_ax"],
                          title=L["chart_P"], **PL)
        st.plotly_chart(fig, use_container_width=True)

    with tab_v:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r_arr, y=v_arr, mode="lines", name="v(r)",
            line=dict(color="#34d399", width=2.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.07)",
            hovertemplate=f"{L['r_ax']}: %{{x:.1f}}<br>{L['v_ax']}: %{{y:.3e}}<extra></extra>",
        ))
        fig.update_layout(xaxis_title=L["r_ax"], yaxis_title=L["v_ax"],
                          yaxis_type="log", title=L["chart_v"], **PL)
        st.plotly_chart(fig, use_container_width=True)

    with tab_r:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r_arr, y=R_arr, mode="lines", name="R(r)",
            line=dict(color="#f87171", width=2.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.07)",
            hovertemplate=f"{L['r_ax']}: %{{x:.1f}}<br>{L['R_ax']}: %{{y:.4e}}<extra></extra>",
        ))
        fig.add_vrect(x0=float(r_arr[0]), x1=float(s.r_pbz),
                      fillcolor="rgba(108,156,245,0.06)",
                      line=dict(color="#6c9cf5", width=1, dash="dot"),
                      annotation_text="ПЗП" if IS_RU else "NWZ",
                      annotation_font_color="#6c9cf5")
        fig.update_layout(xaxis_title=L["r_ax"], yaxis_title=L["R_ax"],
                          title=L["chart_R"], **PL)
        st.plotly_chart(fig, use_container_width=True)

    # ── ТЕПЛОВАЯ КАРТА ─────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-head">🗺️ {L["hmap_title"]}</div>', unsafe_allow_html=True)
    st.caption(L["hmap_note"])

    def _ring_color(t):
        t = max(0.0, min(1.0, t))
        if t < 0.33:
            tt = t / 0.33
            return f"rgba({int(30-tt*4)},{int(58+tt*49)},{int(95-tt*21)},0.88)"
        elif t < 0.66:
            tt = (t - 0.33) / 0.33
            return f"rgba({int(26+tt*154)},{int(107-tt*24)},{int(74-tt*65)},0.88)"
        else:
            tt = (t - 0.66) / 0.34
            return f"rgba({int(180+tt*59)},{int(83-tt*15)},{int(9+tt*59)},0.88)"

    MAX_RINGS = 40
    step_r = max(1, len(r_arr) // MAX_RINGS)
    r_ds   = r_arr[::step_r]
    R_ds   = R_arr[::step_r]
    R_mx   = float(R_ds.max()) if R_ds.max() > 0 else 1.0
    widths = np.diff(r_ds, prepend=0.0)
    N_SEC  = 36
    d_th   = 360.0 / N_SEC
    thetas = np.arange(0, 360, d_th)

    fig_hm = go.Figure()
    for rv, Rv, w in zip(r_ds, R_ds, widths):
        clr = _ring_color(Rv / R_mx)
        fig_hm.add_trace(go.Barpolar(
            r=[rv] * N_SEC, theta=thetas, width=[d_th] * N_SEC, base=[rv - w] * N_SEC,
            marker=dict(color=clr, line=dict(width=0)),
            hovertemplate=f"r = {rv:.1f} м<br>R = {Rv:.3e} {L['kpi_R_unit']}<extra></extra>",
            showlegend=False,
        ))
    # Граница ПЗП
    fig_hm.add_trace(go.Barpolar(
        r=[s.r_pbz] * N_SEC, theta=thetas,
        width=[d_th] * N_SEC, base=[0] * N_SEC,
        marker=dict(color="rgba(0,0,0,0)", line=dict(color="#6c9cf5", width=1.5)),
        hoverinfo="skip", showlegend=False,
    ))
    # Скважина
    fig_hm.add_trace(go.Scatterpolar(
        r=[0], theta=[0], mode="markers",
        marker=dict(size=10, color="white", symbol="circle"),
        hovertemplate="Скважина<extra></extra>" if IS_RU else "Wellbore<extra></extra>",
        showlegend=False,
    ))
    fig_hm.update_layout(
        polar=dict(
            bgcolor="#0d1017",
            radialaxis=dict(
                visible=True, showgrid=True, gridcolor="#1e2230",
                tickfont=dict(color="#6b7280", size=9, family="JetBrains Mono"),
                ticksuffix=" m", range=[0, float(r_ds[-1]) * 1.05],
            ),
            angularaxis=dict(showgrid=False, showticklabels=False),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans,sans-serif", color="#9ca3af"),
        height=460, margin=dict(l=40, r=40, t=40, b=40),
        hoverlabel=dict(bgcolor="#1e2230", bordercolor="#2d3748",
                        font=dict(family="JetBrains Mono", color="#e8eaf0")),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ────────────────────────────────────────────
    st.markdown(f'<div class="sec-head">🔬 {L["sens_title"]}</div>', unsafe_allow_html=True)
    st.caption(L["sens_note"])

    if st.button(L["sens_btn"]):
        with st.spinner("Расчёт..." if IS_RU else "Calculating..."):
            base_kw = dict(
                T=s.T, pH=s.pH, C_Ca=s.C_Ca, C_HCO3=s.C_HCO3,
                C_NaK=s.C_NaK, C_Cl=s.C_Cl,
                m=s.m, rock_type=L["rock_val"][s.rock_idx], k=s.k,
                Q_w=s.Q_w, R_e=s.R_e, r_w=s.r_w, h=s.h,
                P_e=s.P_e, P_w=s.P_w, t=s.t, n_points=60,
            )

            def _sweep(param, values):
                out = []
                for v in values:
                    try:
                        kw = {**base_kw, param: v}
                        r2 = run_model(**kw)
                        out.append(float(r2["R_arr"][0]))
                    except Exception:
                        out.append(0.0)
                return out

            pH_r = np.linspace(5.5, 10.0, 50)
            T_r  = np.linspace(20,  120,   50)
            Pw_r = np.linspace(max(0.5, s.P_w * 0.4), s.P_e * 0.95, 50)

            R_pH = _sweep("pH",  pH_r)
            R_T  = _sweep("T",   T_r)
            R_Pw = _sweep("P_w", Pw_r)

        ax_s = dict(gridcolor="#1e2230", linecolor="#1e2230",
                    zerolinecolor="#2d3748", tickfont=dict(color="#9ca3af"))
        fig_s = make_subplots(rows=1, cols=3,
            subplot_titles=[L["sens_pH"], L["sens_T"], L["sens_Pw"]])

        for col, x, y, clr in [
            (1, pH_r, R_pH, "#6c9cf5"),
            (2, T_r,  R_T,  "#a78bfa"),
            (3, Pw_r, R_Pw, "#34d399"),
        ]:
            fig_s.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color=clr, width=2.5),
                fill="tozeroy", fillcolor=clr.replace(")", ",0.08)").replace("#6c9cf5","rgba(108,156,245").replace("#a78bfa","rgba(167,139,250").replace("#34d399","rgba(52,211,153"),
                hovertemplate=f"%{{x:.2f}} → R=%{{y:.3e}}<extra></extra>",
            ), row=1, col=col)

        fig_s.update_layout(
            height=340, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1017",
            font=dict(family="DM Sans,sans-serif", color="#9ca3af"),
            margin=dict(l=40, r=20, t=50, b=40),
            hoverlabel=dict(bgcolor="#1e2230",
                            font=dict(family="JetBrains Mono", color="#e8eaf0")),
        )
        for ax in ["xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"]:
            fig_s.update_layout(**{ax: ax_s})
        st.plotly_chart(fig_s, use_container_width=True)

    # ── НАКОПЛЕНИЕ M(t) ────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-head">⏱ {L["acc_title"]}</div>', unsafe_allow_html=True)
    years = st.slider(L["acc_years"], 1, 20, 5, key="acc_slider")

    if st.button(L["acc_btn"]):
        t_pts = np.linspace(0, years * 365, 80)

        # Интегралы по зонам
        mask_pbz = r_arr <= max(s.r_pbz, s.r_w * 2)
        if mask_pbz.sum() >= 2:
            intg_pbz = float(np.trapezoid(
                R_arr[mask_pbz] * 2 * math.pi * r_arr[mask_pbz] * s.h,
                r_arr[mask_pbz]
            ))
        else:
            intg_pbz = float(R_arr[0]) * math.pi * (s.r_pbz**2 - s.r_w**2) * s.h

        intg_all = float(np.trapezoid(R_arr * 2 * math.pi * r_arr * s.h, r_arr))

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=t_pts / 365, y=t_pts * intg_pbz,
            mode="lines", name=L["acc_pbz"],
            line=dict(color="#6c9cf5", width=2.5),
            fill="tozeroy", fillcolor="rgba(108,156,245,0.08)",
        ))
        fig_acc.add_trace(go.Scatter(
            x=t_pts / 365, y=t_pts * intg_all,
            mode="lines", name=L["acc_all"],
            line=dict(color="#f87171", width=1.5, dash="dot"),
        ))
        fig_acc.update_layout(
            xaxis_title="t (лет)" if IS_RU else "t (years)",
            yaxis_title="M (кг)" if IS_RU else "M (kg)",
            title="M(t)",
            legend=dict(font=dict(color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
            **PL,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # ── РЕКОМЕНДАЦИИ ───────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-head">💡 {L["rec_title"]}</div>', unsafe_allow_html=True)
    recs, _ = _recommendations(SI_arr, R_arr, M_pbz)
    if not recs:
        st.markdown(f'<div class="rec-card ok">{L["rec_ok"]}</div>', unsafe_allow_html=True)
    else:
        for cls, txt in recs:
            st.markdown(f'<div class="rec-card {cls}">{txt}</div>', unsafe_allow_html=True)

    # ── ДЕТАЛИ РАСЧЁТА ─────────────────────────────────────────────────────
    with st.expander(f"🔎 {L['det_title']}"):
        dc1, dc2 = st.columns(2)
        with dc1:
            st.metric(L["det_SIe"],  f"{res.get('SI_e', 0):+.4f}")
            st.metric(L["det_SIrw"], f"{float(SI_arr[0]):+.4f}")
            st.metric(L["det_Rrw"],  f"{float(R_arr[0]):.4e}")
        with dc2:
            st.metric(L["det_Rmax"], f"{R_max:.4e}")
            st.metric(L["det_Mt"],   f"{M_t:.1f} {'кг' if IS_RU else 'kg'}")
            st.metric(L["det_Mpbz"], f"{M_pbz:.2f} {'кг' if IS_RU else 'kg'}")

