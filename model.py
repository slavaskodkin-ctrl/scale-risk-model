"""
Гидродинамическая модель прогнозирования гетерогенной нуклеации
и риска солеотложения CaCO₃ в поровом пространстве коллектора.

Версия 2.2 — физически откалиброванная модель.

Три ключевых исправления по сравнению с v1:
──────────────────────────────────────────────────────────────────
1. КИНЕТИКА R(r): логарифмическая форма движущей силы SI²
   вместо (S-1)². При SI=2.75: (S-1)²=315000 vs SI²=7.6 — разница
   в 40000 раз. Логарифмическая кинетика стандартна для пластовых
   расчётов при высоком пересыщении (Raines & Dewers, 1997).

2. ЭФФЕКТИВНАЯ КОНСТАНТА K_EFF = 1e-12 моль/(м²·с).
   Лабораторный k_kin (Plummer & Busenberg, 1982) = 5e-9 — на 3
   порядка выше реального пластового из-за органических ингибиторов,
   Mg²⁺, ограниченного массопереноса.

3. МАССА M(t): реакционно-транспортный подход.
   Интеграл по всему пласту завышает M, т.к. не учитывает истощение
   Ca²⁺ по мере осаждения. Используем:
       M(t) = Q_w · t · c_Ca · (M_CaCO3/M_Ca) · η
   где η = tanh(SI_e) · 0.10 — доля выпавшего Ca²⁺.
   η растёт с пересыщением, не превышает ~10% (промысловые данные,
   Moghadasi et al., 2004). R(r) используется для ПРОФИЛЯ риска,
   M(t) — для оценки суммарной массы.

Единицы:
    SI_arr  [-]            — безразмерный
    R_arr   [кг/(м³·сут)] — скорость осаждения
    M_t     [кг]          — суммарная масса за период
"""

import math
import numpy as np

# ─── Молярные массы (г/моль) ────────────────────────────────────────────────
M_Ca    = 40.078
M_HCO3  = 61.016
M_Na    = 22.990
M_K     = 39.098
M_Cl    = 35.453
M_CaCO3 = 100.09

# ─── Константы модели ───────────────────────────────────────────────────────
K_EFF        = 1.0e-12  # эффективная пластовая кинетическая константа [моль/(м²·с)]
N_ORDER      = 2        # порядок логарифмической кинетики
ETA_FACTOR   = 0.10     # максимальная доля Ca²⁺, выпадающего в осадок (эмпирически ~10%)
LAMBDA_SENS  = 0.1      # чувствительность SI к давлению [МПа⁻¹]
ALPHA        = 0.5      # коэффициент подавления барьера нуклеации
BETA         = 5.0e4    # гидродинамический коэффициент [с/м]
C_KOZENY     = 5.0      # коэффициент Козени–Кармана
SI_THRESHOLD = 0.001    # порог SI для начала осаждения

A_DH = 0.509
B_DH = 0.328
A_ION = {"Ca": 6.0, "CO3": 5.0, "Na": 4.0, "K": 3.0, "Cl": 3.5, "HCO3": 4.0}

# ─── 4. ХИМИЧЕСКИЙ БЛОК ─────────────────────────────────────────────────────

def _convert_concentrations(C_Ca, C_HCO3, C_NaK, C_Cl):
    """4.1 мг/л → моль/л"""
    return {
        "Ca":   C_Ca   / (1000.0 * M_Ca),
        "HCO3": C_HCO3 / (1000.0 * M_HCO3),
        "Na":   (C_NaK / 2.0) / (1000.0 * M_Na),
        "K":    (C_NaK / 2.0) / (1000.0 * M_K),
        "Cl":   C_Cl   / (1000.0 * M_Cl),
    }


def _calc_Ksp(T_C):
    """4.2 Ksp(T) — уравнение ван'т-Гоффа"""
    T_K = T_C + 273.15
    return max(10.0**(-8.48) * math.exp((9000.0 / 8.314) * (1.0/T_K - 1.0/298.15)), 1e-30)


def _calc_CO3_concentration(c_HCO3, pH, Ksp):
    """4.2 c(CO₃²⁻) из равновесия карбонатной системы"""
    if c_HCO3 <= 0.0:
        return 0.0
    return max(c_HCO3 * 10.0**(pH + math.log10(Ksp)), 0.0)


def _calc_ionic_strength(c_ions):
    """4.3 Ионная сила [моль/л]: I = 0.5 · Σ(c_i · z_i²)"""
    charges = {"Ca": 2, "CO3": 2, "HCO3": 1, "Na": 1, "K": 1, "Cl": 1}
    return max(0.5 * sum(c_ions.get(k, 0.0) * z**2 for k, z in charges.items()), 1e-15)


def _calc_activity_coeff(z, a_i, I):
    """4.4 Коэффициент активности (расширенный Дебай–Хюккель)"""
    sqrt_I = math.sqrt(I)
    denom  = 1.0 + B_DH * a_i * sqrt_I
    return 1.0 if abs(denom) < 1e-30 else 10.0**(-A_DH * z**2 * sqrt_I / denom)


def _calc_activities(c_ions, I):
    """4.5 Активности Ca²⁺ и CO₃²⁻"""
    return (_calc_activity_coeff(2, A_ION["Ca"],  I) * c_ions["Ca"],
            _calc_activity_coeff(2, A_ION["CO3"], I) * c_ions["CO3"])


def _calc_SI_base(a_Ca, a_CO3, Ksp):
    """4.6 SI_e = log10(a_Ca · a_CO3 / Ksp)"""
    p = a_Ca * a_CO3
    return math.log10(p / Ksp) if (p > 0.0 and Ksp > 0.0) else -999.0


# ─── 5. БЛОК ФЕС ────────────────────────────────────────────────────────────

def _calc_wettability(rock_type):
    """
    5.1 Угол смачиваемости и γ_cl [Дж/м²].
    Песчаник : cos(θ)=−0.30  →  f(θ)=0.718  (высокий барьер нуклеации)
    Карбонат : cos(θ)=+0.25  →  f(θ)=0.492  (низкий барьер — осадок легче)
    """
    gamma_cl = 0.1
    if rock_type.lower() in ("sandstone", "песчаник"):
        cos_theta = (0.04 - 0.07) / gamma_cl
    elif rock_type.lower() in ("carbonate", "карбонат"):
        cos_theta = (0.04 - 0.015) / gamma_cl
    else:
        raise ValueError(f"Неизвестный тип коллектора: '{rock_type}'. "
                         "Допустимые значения: 'sandstone' / 'carbonate'.")
    return max(-1.0, min(1.0, cos_theta)), gamma_cl


def _calc_geometric_factor(cos_theta):
    """5.2 f(θ) = (2+cosθ)(1−cosθ)²/4 — геометрический фактор CNT"""
    return (2.0 + cos_theta) * (1.0 - cos_theta)**2 / 4.0


def _calc_specific_surface(m, k):
    """
    5.3 Удельная поверхность пор S_v [м²/м³] — Козени–Карман в СИ.
    1 мД = 9.869·10⁻¹⁶ м²
    При k=50 мД, m=0.20: S_v ≈ 225 000 м²/м³.
    """
    if not (0.0 < m < 1.0):
        raise ValueError(f"Пористость m={m} вне диапазона (0, 1).")
    if k <= 0.0:
        raise ValueError(f"Проницаемость k={k} должна быть > 0.")
    k_m2 = k * 9.869e-16
    return math.sqrt(m**3 / (C_KOZENY * k_m2 * (1.0 - m)**2))


# ─── 6. ГИДРОДИНАМИЧЕСКИЙ БЛОК ──────────────────────────────────────────────

def _calc_darcy_velocity(Q_w, r, h):
    """6.1 Скорость Дарси u(r) [м/с]: u = Q_w / (2π·r·h·86400)"""
    return Q_w / (2.0 * math.pi * r * h * 86400.0)


def _calc_pressure(r, P_e, P_w, R_e, r_w):
    """6.2 Давление P(r) [МПа] при плоско-радиальной фильтрации"""
    r = max(r_w, min(r, R_e))
    ln_outer = math.log(R_e / r_w)
    return P_e if abs(ln_outer) < 1e-30 else \
           P_e - (P_e - P_w) / ln_outer * math.log(R_e / r)


def _calc_SI_local(SI_e, P_e, P_r):
    """6.3 SI(r) = SI_e + λ·(P_e − P(r))"""
    return SI_e + LAMBDA_SENS * (P_e - P_r)


def _calc_nucleation_barrier(SI_r, gamma_cl, f_theta):
    """
    6.4–6.6 Барьер гетерогенной нуклеации (классическая теория нуклеации, CNT).
        B_het = γ_cl³ · f(θ) / (ln S)²

    γ_cl³ — когезия кристалла (межфазное натяжение кристалл–жидкость)
    f(θ)  — адгезия к поверхности (смачиваемость)
    При SI ≤ 0 → нуклеация невозможна.
    """
    if SI_r <= 0.0:
        return float("inf")
    ln_S = math.log(10.0) * SI_r
    return float("inf") if abs(ln_S) < 1e-30 else (gamma_cl**3 * f_theta) / ln_S**2


def _calc_flow_factor(v_r):
    """6.7 F_flow = 1/(1+β·v) — гидродинамическое подавление кристаллизации"""
    return 1.0 / (1.0 + BETA * v_r)


# ─── 7. СКОРОСТЬ СОЛЕОТЛОЖЕНИЯ ──────────────────────────────────────────────

def _calc_deposition_rate(SI_r, Sv, gamma_cl, f_theta, F_flow):
    """
    7. Локальная скорость солеотложения R(r) [кг/(м³·сут)].

    Формула v2.2:
        Движущая сила (логарифмическая): SI²
        Барьер нуклеации:               W = exp(−α·B_het)
        Гидродинамика:                  F_flow
        R = K_EFF · S_v · SI² · W · F_flow · (M_CaCO3/1000) · 86400

    R используется для построения ПРОФИЛЯ риска R(r), показывающего
    где в пласте концентрируется осаждение.
    """
    if SI_r <= SI_THRESHOLD:
        return 0.0
    B_het = _calc_nucleation_barrier(SI_r, gamma_cl, f_theta)
    W     = math.exp(-ALPHA * B_het) if B_het < 700.0 else 0.0
    R_kin = K_EFF * Sv * (SI_r**N_ORDER) * W * F_flow * (M_CaCO3 / 1000.0) * 86400.0
    return max(0.0, R_kin)


# ─── 8. СУММАРНАЯ МАССА ─────────────────────────────────────────────────────

def _calc_total_mass(Q_w, t, c_Ca_mol_L, SI_e):
    """
    8. Суммарная масса осадка M(t) [кг] — реакционно-транспортный подход.

    M(t) = Q_w · t · c_Ca[кг/м³] · (M_CaCO3/M_Ca) · η(SI)

    где:
        Q_w · t · c_Ca · M_CaCO3/M_Ca  — масса CaCO₃ при 100% осаждении Ca²⁺
        η(SI) = tanh(SI_e) · ETA_FACTOR — доля выпавшего Ca²⁺

    η растёт с пересыщением (больше SI → больше осадка), не превышает
    ETA_FACTOR ≈ 10% (характерно для пластовых условий при нормальной
    эксплуатации без обработки ингибитором).

    Физический смысл η: пересыщенный раствор не осаждает Ca²⁺ мгновенно —
    реальная доля осаждения определяется временем пребывания воды в пласте,
    наличием центров нуклеации и кинетикой роста кристаллов.
    """
    if SI_e <= 0.0:
        return 0.0
    c_Ca_kg_m3 = c_Ca_mol_L * 1000.0 * (M_Ca / 1000.0)   # кг/м³
    M_max      = Q_w * t * c_Ca_kg_m3 * (M_CaCO3 / M_Ca)  # кг (100% предел)
    eta        = math.tanh(max(SI_e, 0.0)) * ETA_FACTOR
    return M_max * eta


# ─── ГЛАВНАЯ ФУНКЦИЯ ────────────────────────────────────────────────────────

def run_model(T, pH, C_Ca, C_HCO3, C_NaK, C_Cl,
              m, rock_type, k,
              Q_w, R_e, r_w, h, P_e, P_w, t,
              n_points=150):
    """
    Запуск модели солеотложения CaCO₃ v2.2.

    Возвращает dict:
        r_arr   [м]             — радиальная сетка
        SI_arr  [-]             — индекс насыщения SI(r)
        R_arr   [кг/(м³·сут)]  — скорость солеотложения R(r)  ← профиль риска
        M_t     [кг]           — масса за t суток             ← итоговое число
        M_max   [кг]           — предел при 100% осаждении Ca²⁺
        eta     [-]             — расчётная доля осаждения
        u_arr   [м/с]          — скорость Дарси u(r)
        v_arr   [м/с]          — поровая скорость v(r)
        P_arr   [МПа]          — давление P(r)
        SI_e    [-]             — базовый индекс насыщения
    """
    # 4. Химия
    c_ions        = _convert_concentrations(C_Ca, C_HCO3, C_NaK, C_Cl)
    Ksp           = _calc_Ksp(T)
    c_ions["CO3"] = _calc_CO3_concentration(c_ions["HCO3"], pH, Ksp)
    I             = _calc_ionic_strength(c_ions)
    a_Ca, a_CO3   = _calc_activities(c_ions, I)
    SI_e          = _calc_SI_base(a_Ca, a_CO3, Ksp)

    # 5. ФЕС
    cos_theta, gamma_cl = _calc_wettability(rock_type)
    f_theta             = _calc_geometric_factor(cos_theta)
    Sv                  = _calc_specific_surface(m, k)

    # 6. Радиальная сетка и профили
    r_arr  = np.logspace(math.log10(r_w), math.log10(R_e), n_points)
    P_arr  = np.array([_calc_pressure(r, P_e, P_w, R_e, r_w) for r in r_arr])
    SI_arr = np.array([_calc_SI_local(SI_e, P_e, Pr)          for Pr in P_arr])
    u_arr  = np.array([_calc_darcy_velocity(Q_w, r, h)        for r in r_arr])
    v_arr  = u_arr / m

    # 7. Профиль R(r)
    R_arr = np.zeros(n_points)
    for i in range(n_points):
        if SI_arr[i] <= SI_THRESHOLD:
            continue
        F_flow   = _calc_flow_factor(v_arr[i])
        R_arr[i] = _calc_deposition_rate(SI_arr[i], Sv, gamma_cl, f_theta, F_flow)

    # 8. Суммарная масса
    M_t  = _calc_total_mass(Q_w, t, c_ions["Ca"], SI_e)
    eta  = math.tanh(max(SI_e, 0.0)) * ETA_FACTOR if SI_e > 0 else 0.0
    M_max = Q_w * t * c_ions["Ca"] * 1000.0 * (M_Ca/1000.0) * (M_CaCO3 / M_Ca)

    return {
        "r_arr":  r_arr,
        "SI_arr": SI_arr,
        "R_arr":  R_arr,
        "M_t":    M_t,
        "M_max":  M_max,
        "eta":    eta,
        "u_arr":  u_arr,
        "v_arr":  v_arr,
        "P_arr":  P_arr,
        "SI_e":   SI_e,
    }


# ─── САМОПРОВЕРКА ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    res = run_model(
        T=60, pH=7.2, C_Ca=800, C_HCO3=400, C_NaK=5000, C_Cl=8000,
        m=0.20, rock_type="sandstone", k=50,
        Q_w=100, R_e=500, r_w=0.1, h=10, P_e=20.0, P_w=15.0, t=365,
    )
    r = res["r_arr"]; SI = res["SI_arr"]; R = res["R_arr"]
    print("=" * 62)
    print("  МОДЕЛЬ CaCO₃ v2.2 — САМОПРОВЕРКА")
    print("=" * 62)
    print(f"  SI_e (равновесный)  = {res['SI_e']:+.4f}")
    print(f"  SI(r_w)             = {SI[0]:+.4f}")
    print(f"  SI(R_e)             = {SI[-1]:+.4f}")
    print(f"  R(r_w)              = {R[0]:.4e} кг/(м³·сут)")
    print(f"  R(R_e)              = {R[-1]:.4e} кг/(м³·сут)")
    print(f"  R_max               = {R.max():.4e} кг/(м³·сут)  при r={r[R.argmax()]:.1f} м")
    print(f"  η (доля осаждения)  = {res['eta']*100:.2f}%")
    print(f"  M_max (100% Ca²⁺)   = {res['M_max']:.1f} кг")
    print(f"  M(365 сут)          = {res['M_t']:.2f} кг  ← итоговый результат")
    print(f"  M(365 сут)          = {res['M_t']/1000:.4f} т")
    print("=" * 62)

