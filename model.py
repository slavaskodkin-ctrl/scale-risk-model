"""
Гидродинамическая модель прогнозирования гетерогенной нуклеации
и риска солеотложения CaCO₃ в поровом пространстве коллектора.

Входные данные:
  Химический блок:
    T        - температура (°C)
    pH       - водородный показатель
    C_Ca     - концентрация Ca²⁺ (мг/л)
    C_HCO3   - концентрация HCO₃⁻ (мг/л)
    C_NaK    - суммарная концентрация Na⁺ + K⁺ (мг/л)
    C_Cl     - концентрация Cl⁻ (мг/л)

  Блок ФЕС коллектора:
    m        - пористость (доли, 0–1)
    rock_type - тип коллектора: "sandstone" | "carbonate"
    k        - фазовая проницаемость по воде (мД)

  Гидродинамический блок:
    Q_w      - дебит воды (м³/сут)
    R_e      - радиус контура питания (м)
    r_w      - радиус скважины (м)
    h        - эффективная толщина пласта (м)
    P_e      - пластовое давление (МПа)
    P_w      - забойное давление (МПа)
    t        - период эксплуатации (сут)
    n_points - число радиальных узлов сетки (по умолчанию 200)

Ключевые выходные данные:
    r_arr  - массив радиусов (м)
    SI_arr - профиль индекса насыщения SI(r)       [-]
    R_arr  - профиль скорости солеотложения R(r)   [т/(м³·сут)]
    M_t    - суммарная масса солеотложений M(t)    [т]
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Молярные массы (г/моль)
# ---------------------------------------------------------------------------
M_Ca   = 40.078
M_HCO3 = 61.016
M_Na   = 22.990
M_K    = 39.098
M_Cl   = 35.453

# ---------------------------------------------------------------------------
# Константы модели
# ---------------------------------------------------------------------------
LAMBDA_SENS  = 0.1      # МПа⁻¹  — чувствительность пересыщения к давлению
ALPHA        = 0.5      # коэффициент энергетического подавления нуклеации
BETA         = 5e4      # гидродинамический коэффициент (с/м)
C_KOZENY     = 5        # коэффициент в формуле удельной поверхности (Козени–Карман)
SI_THRESHOLD = 0.001    # порог: при SI ≤ порога считаем R = 0

# Коэффициенты расширенного уравнения Дебая-Хюккеля
A_DH = 0.509
B_DH = 0.328

# Эффективные радиусы ионов (Å) для уравнения Дебая-Хюккеля
A_ION = {
    "Ca":   6.0,
    "CO3":  5.0,
    "Na":   4.0,
    "K":    3.0,
    "Cl":   3.5,
    "HCO3": 4.0,
}

# ---------------------------------------------------------------------------
# 4. ХИМИЧЕСКИЙ БЛОК
# ---------------------------------------------------------------------------

def _convert_concentrations(C_Ca, C_HCO3, C_NaK, C_Cl):
    """
    4.1 Перевод концентраций из мг/л в моль/л.
    c_i = C_i / (1000 * M_i)
    C_NaK делится поровну между Na⁺ и K⁺ для расчёта ионной силы.
    """
    c_Ca   = C_Ca   / (1000 * M_Ca)
    c_HCO3 = C_HCO3 / (1000 * M_HCO3)
    c_Na   = (C_NaK / 2) / (1000 * M_Na)
    c_K    = (C_NaK / 2) / (1000 * M_K)
    c_Cl   = C_Cl   / (1000 * M_Cl)
    return {"Ca": c_Ca, "HCO3": c_HCO3, "Na": c_Na, "K": c_K, "Cl": c_Cl}


def _calc_Ksp(T_C):
    """
    4.2 Произведение растворимости CaCO₃ как функция температуры.
    T_K = T_C + 273.15
    Ksp = 10^(-8.48) * exp( (9000 / R) * (1/T_K - 1/298.15) )
    где R = 8.314 Дж/(моль·К) — универсальная газовая постоянная.
    """
    T_K = T_C + 273.15
    Ksp = 10**(-8.48) * math.exp((9000 / 8.314) * (1 / T_K - 1 / 298.15))
    return max(Ksp, 1e-30)


def _calc_CO3_concentration(c_HCO3, pH, Ksp):
    """
    4.2 Концентрация CO₃²⁻ (моль/л) из гетерогенного равновесия.
    c_CO3 = c_HCO3 * 10^(pH + log10(Ksp))
    """
    if c_HCO3 <= 0:
        return 0.0
    c_CO3 = c_HCO3 * 10 ** (pH + math.log10(Ksp))
    return max(c_CO3, 0.0)


def _calc_ionic_strength(c_ions):
    """
    4.3 Ионная сила (моль/л).
    I = 0.5 * Σ(c_i * z_i²)
    Заряды: Ca²⁺=2, CO₃²⁻=2, HCO₃⁻=1, Na⁺=1, K⁺=1, Cl⁻=1
    """
    charges = {"Ca": 2, "CO3": 2, "HCO3": 1, "Na": 1, "K": 1, "Cl": 1}
    I = 0.5 * sum(c_ions.get(ion, 0.0) * z**2 for ion, z in charges.items())
    return max(I, 1e-15)


def _calc_activity_coeff(z, a_i, I):
    """
    4.4 Коэффициент активности по расширенному уравнению Дебая-Хюккеля.
    log10(γ) = −A * z² * √I / (1 + B * a_i * √I)
    """
    sqrt_I = math.sqrt(I)
    denom = 1.0 + B_DH * a_i * sqrt_I
    if abs(denom) < 1e-30:
        return 1.0
    return 10 ** (- A_DH * z**2 * sqrt_I / denom)


def _calc_activities(c_ions, I):
    """
    4.5 Активности Ca²⁺ и CO₃²⁻.
    a_i = γ_i * c_i
    """
    gamma_Ca  = _calc_activity_coeff(2, A_ION["Ca"],  I)
    gamma_CO3 = _calc_activity_coeff(2, A_ION["CO3"], I)
    return gamma_Ca * c_ions["Ca"], gamma_CO3 * c_ions["CO3"]


def _calc_SI_base(a_Ca, a_CO3, Ksp):
    """
    4.6 Базовый индекс насыщения (равновесный).
    SI_e = log10( a_Ca * a_CO3 / Ksp )
    """
    product = a_Ca * a_CO3
    if product <= 0 or Ksp <= 0:
        return -999.0
    return math.log10(product / Ksp)


# ---------------------------------------------------------------------------
# 5. БЛОК ФЕС КОЛЛЕКТОРА
# ---------------------------------------------------------------------------

def _calc_wettability(rock_type):
    """
    5.1 Косинус угла смачиваемости и межфазное натяжение жидкость–кристалл.
    cos(θ) = (γ_rl − γ_cr) / γ_cl,  γ_cl = 0.1 Дж/м²
    Песчаник : γ_rl=0.04, γ_cr=0.07  → cos(θ) = −0.30
    Карбонат : γ_rl=0.04, γ_cr=0.015 → cos(θ) = +0.25
    """
    gamma_cl = 0.1
    if rock_type.lower() in ("sandstone", "песчаник"):
        cos_theta = (0.04 - 0.07) / gamma_cl   # = −0.30
    elif rock_type.lower() in ("carbonate", "карбонат"):
        cos_theta = (0.04 - 0.015) / gamma_cl  # = +0.25
    else:
        raise ValueError(
            f"Неизвестный тип коллектора: '{rock_type}'. "
            "Допустимые значения: 'sandstone' / 'carbonate'."
        )
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return cos_theta, gamma_cl


def _calc_geometric_factor(cos_theta):
    """
    5.2 Геометрический фактор гетерогенной нуклеации.
    f(θ) = (2 + cos θ) * (1 − cos θ)² / 4
    """
    return (2 + cos_theta) * (1 - cos_theta)**2 / 4


def _calc_specific_surface(m, k):
    """
    5.2 Удельная поверхность пор S_v (м²/м³) — формула Козени–Кармана.
    S_v = sqrt( m³ / (C * k * (1−m)² * 1e-15) )
    C = 5, k в мД (1 мД = 1e-15 м² в данном контексте формулы).
    """
    if not (0 < m < 1):
        raise ValueError(f"Пористость m={m} должна быть в диапазоне (0, 1).")
    if k <= 0:
        raise ValueError(f"Проницаемость k={k} должна быть > 0.")
    denom = C_KOZENY * k * (1 - m)**2 * 1e-15
    return math.sqrt(m**3 / denom)


# ---------------------------------------------------------------------------
# 6. ГИДРОДИНАМИЧЕСКИЙ БЛОК
# ---------------------------------------------------------------------------

def _calc_darcy_velocity(Q_w, r, h):
    """
    6.1 Скорость фильтрации Дарси u(r) (м/с).
    u(r) = Q_w / (2π * r * h * 86400)
    Q_w [м³/сут] → делим на 86400 для перевода в м³/с.
    """
    if r <= 0 or h <= 0:
        raise ValueError("Радиус r и толщина h должны быть > 0.")
    return Q_w / (2 * math.pi * r * h * 86400)


def _calc_pressure(r, P_e, P_w, R_e, r_w):
    """
    6.2 Давление в точке r (МПа) при плоско-радиальной фильтрации.
    P(r) = P_e − (P_e − P_w) / ln(R_e/r_w) * ln(R_e/r)
    """
    r = max(r_w, min(r, R_e))
    ln_outer = math.log(R_e / r_w)
    if abs(ln_outer) < 1e-30:
        return P_e
    return P_e - (P_e - P_w) / ln_outer * math.log(R_e / r)


def _calc_SI_local(SI_e, P_e, P_r):
    """
    6.3 Локальный индекс насыщения с поправкой на падение давления.
    SI(r) = SI_e + λ * (P_e − P(r)),  λ = 0.1 МПа⁻¹
    """
    return SI_e + LAMBDA_SENS * (P_e - P_r)


def _calc_nucleation_barrier(SI_r, gamma_cl, f_theta):
    """
    6.4–6.6 Энергетический барьер гетерогенной нуклеации.

    S(r)      = 10^SI(r)                        — степень пересыщения
    B_hom(r)  = γ_cl³ / (ln S(r))²             — гомогенный барьер
    B_het(r)  = B_hom(r) * f(θ)                — гетерогенный барьер

    При SI ≤ 0 барьер бесконечен (нуклеация невозможна).
    """
    if SI_r <= 0:
        return float("inf")
    ln_S = math.log(10) * SI_r   # ln(10^SI) = SI * ln(10)
    if abs(ln_S) < 1e-30:
        return float("inf")
    B_hom = gamma_cl**3 / ln_S**2
    return B_hom * f_theta


def _calc_flow_factor(v_r):
    """
    6.7 Гидродинамический фактор подавления кристаллизации.
    F_flow = 1 / (1 + β * v(r)),  β = 5·10⁴ с/м
    """
    return 1.0 / (1.0 + BETA * v_r)


# ---------------------------------------------------------------------------
# 7. ФУНКЦИЯ РИСКА СОЛЕОТЛОЖЕНИЯ
# ---------------------------------------------------------------------------

def _calc_deposition_rate(SI_r, Sv, k, B_het, F_flow):
    """
    7. Локальная скорость солеотложения R(r) [т/(м³·сут)].

    Если SI(r) ≤ SI_THRESHOLD → R = 0 (раствор не пересыщен).
    Иначе:
        R(r) = (S_v / sqrt(k * 1e-15)) * exp(−α * B_het) * F_flow / (86400 * 1000)

    Делитель 86400 * 1000 переводит из [кг/(м³·с)] в [т/(м³·сут)].
    """
    if SI_r <= SI_THRESHOLD:
        return 0.0
    k_m2 = k * 1e-15   # мД → м²
    if k_m2 <= 0:
        return 0.0
    rate = (Sv / math.sqrt(k_m2)) * math.exp(-ALPHA * B_het) * F_flow / (86400 * 1000)
    return max(rate, 0.0)


def _calc_total_mass(r_arr, R_arr, h, t):
    """
    7. Суммарная масса солеотложений за время t (т).
    M(t) = t * ∫[r_w..R_e] R(r) * 2π * r * h dr
    Численное интегрирование методом трапеций.
    """
    integrand = R_arr * 2 * math.pi * r_arr * h
    return t * np.trapezoid(integrand, r_arr)


# ---------------------------------------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ
# ---------------------------------------------------------------------------

def run_model(
    # --- Химический блок ---
    T, pH, C_Ca, C_HCO3, C_NaK, C_Cl,
    # --- ФЕС ---
    m, rock_type, k,
    # --- Гидродинамика ---
    Q_w, R_e, r_w, h, P_e, P_w, t,
    # --- Параметры сетки ---
    n_points=200,
):
    """
    Запуск полной модели солеотложения CaCO₃.

    Параметры
    ----------
    T         : float  — температура пласта (°C)
    pH        : float  — pH пластовой воды
    C_Ca      : float  — концентрация Ca²⁺ (мг/л)
    C_HCO3    : float  — концентрация HCO₃⁻ (мг/л)
    C_NaK     : float  — суммарная концентрация Na⁺ + K⁺ (мг/л)
    C_Cl      : float  — концентрация Cl⁻ (мг/л)
    m         : float  — пористость (доли, 0–1)
    rock_type : str    — тип коллектора: 'sandstone' или 'carbonate'
    k         : float  — проницаемость по воде (мД)
    Q_w       : float  — дебит воды (м³/сут)
    R_e       : float  — радиус контура питания (м)
    r_w       : float  — радиус скважины (м)
    h         : float  — эффективная толщина пласта (м)
    P_e       : float  — пластовое давление (МПа)
    P_w       : float  — забойное давление (МПа)
    t         : float  — период эксплуатации (сут)
    n_points  : int    — число узлов радиальной сетки (по умолчанию 200)

    Возвращает
    ----------
    dict с ключевыми результатами:
        "r_arr"  : np.ndarray — радиусы (м)
        "SI_arr" : np.ndarray — индекс насыщения SI(r)      [-]
        "R_arr"  : np.ndarray — скорость солеотложения R(r) [т/(м³·сут)]
        "M_t"    : float      — масса солеотложений M(t)    [т]
    """

    # --- 4.1 Перевод концентраций ---
    c_ions = _convert_concentrations(C_Ca, C_HCO3, C_NaK, C_Cl)

    # --- 4.2 Ksp(T) и c(CO₃²⁻) ---
    Ksp          = _calc_Ksp(T)
    c_ions["CO3"] = _calc_CO3_concentration(c_ions["HCO3"], pH, Ksp)

    # --- 4.3 Ионная сила ---
    I = _calc_ionic_strength(c_ions)

    # --- 4.5 Активности Ca²⁺ и CO₃²⁻ ---
    a_Ca, a_CO3 = _calc_activities(c_ions, I)

    # --- 4.6 Базовый индекс насыщения ---
    SI_e = _calc_SI_base(a_Ca, a_CO3, Ksp)

    # --- 5.1–5.2 Параметры ФЕС ---
    cos_theta, gamma_cl = _calc_wettability(rock_type)
    f_theta             = _calc_geometric_factor(cos_theta)
    Sv                  = _calc_specific_surface(m, k)

    # --- 6. Радиальная сетка (логарифмически равномерная — сгущение у скважины) ---
    r_arr = np.logspace(math.log10(r_w), math.log10(R_e), n_points)

    # --- Профили давления и SI(r) ---
    P_arr  = np.array([_calc_pressure(r, P_e, P_w, R_e, r_w) for r in r_arr])
    SI_arr = np.array([_calc_SI_local(SI_e, P_e, P_r) for P_r in P_arr])

    # --- Скорости фильтрации ---
    u_arr = np.array([_calc_darcy_velocity(Q_w, r, h) for r in r_arr])
    v_arr = u_arr / m   # истинная скорость в поровом пространстве

    # --- 7. Профиль скорости солеотложения R(r) ---
    R_arr = np.zeros(n_points)
    for i in range(n_points):
        if SI_arr[i] <= SI_THRESHOLD:
            continue    # R[i] = 0 уже задан
        B_het    = _calc_nucleation_barrier(SI_arr[i], gamma_cl, f_theta)
        F_flow   = _calc_flow_factor(v_arr[i])
        R_arr[i] = _calc_deposition_rate(SI_arr[i], Sv, k, B_het, F_flow)

    # --- 7. Суммарная масса M(t) ---
    M_t = _calc_total_mass(r_arr, R_arr, h, t)

    return {
        "r_arr":  r_arr,
        "SI_arr": SI_arr,
        "R_arr":  R_arr,
        "M_t":    M_t,
    }


# ---------------------------------------------------------------------------
# ПРИМЕР ЗАПУСКА
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_model(
        # Химический блок
        T=60,       pH=7.2,
        C_Ca=800,   C_HCO3=400,   C_NaK=5000,   C_Cl=8000,
        # ФЕС
        m=0.20,  rock_type="sandstone",  k=50,
        # Гидродинамика
        Q_w=100,  R_e=500,  r_w=0.1,  h=10,
        P_e=20.0,  P_w=15.0,  t=365,
        n_points=200,
    )

    r   = results["r_arr"]
    SI  = results["SI_arr"]
    R   = results["R_arr"]
    M_t = results["M_t"]

    print("=" * 58)
    print("  РЕЗУЛЬТАТЫ МОДЕЛИ СОЛЕОТЛОЖЕНИЯ CaCO₃")
    print("=" * 58)

    # --- SI(r) ---
    print("\n  SI(r) — индекс насыщения:")
    print(f"    При r = r_w = {r[0]:.2f} м  :  SI = {SI[0]:.4f}")
    print(f"    При r = R_e = {r[-1]:.1f} м  :  SI = {SI[-1]:.4f}")
    risk_mask = SI > 0
    if risk_mask.any():
        print(f"    Зона риска (SI > 0): r = {r[risk_mask][0]:.2f} – {r[risk_mask][-1]:.1f} м")
    else:
        print("    Риска солеотложения нет (SI ≤ 0 везде).")

    # --- R(r) ---
    print("\n  R(r) — скорость солеотложения [т/(м³·сут)]:")
    if R.max() > 0:
        idx_max = R.argmax()
        print(f"    Максимум: R = {R[idx_max]:.4e}  при r = {r[idx_max]:.2f} м")
        print(f"    При r = r_w = {r[0]:.2f} м  :  R = {R[0]:.4e}")
        print(f"    При r = R_e = {r[-1]:.1f} м  :  R = {R[-1]:.4e}")
    else:
        print("    R = 0 везде (нет пересыщения).")

    # --- M(t) ---
    print(f"\n  M(t) — масса солеотложений за t = 365 сут:")
    print(f"    M = {M_t:.4e} т")
    print("=" * 58)
