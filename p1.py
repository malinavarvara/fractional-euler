import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 1. Функция Миттаг-Леффлера для точного решения
def mittag_leffler(alpha, beta, z, tol=1e-15, max_terms=1000):
    """
    Вычисление функции Миттаг-Леффлера с контролем точности.
    z: аргумент (в нашем случае t^alpha)
    tol: точность (критерий остановки)
    """
    result = 0.0
    for k in range(max_terms):
        # Вычисляем текущий член ряда
        term = z**k / gamma(alpha * k + beta)
        
        result += term
        
        # Если текущий член стал пренебрежимо мал, выходим
        if abs(term) < tol and k > 5: # k > 5 чтобы не выйти на первых малых членах
            break
            
    return result

def analytical_solution(t, alpha, y0):
    y_true = np.zeros_like(t)
    for i, val in enumerate(t):
        term1 = y0 * mittag_leffler(alpha, 1, val**alpha)
        term2 = (val**alpha) * mittag_leffler(alpha, alpha + 1, val**alpha)
        y_true[i] = term1 + term2
    return y_true

# 2. Расчет метрик погрешности согласно изображениям
def calculate_metrics(y_num, y_true):
    # Максимальная абсолютная погрешность
    e_max = np.max(np.abs(y_num - y_true))
    # Среднеквадратическая погрешность (RMSE)
    rmse = np.sqrt(np.mean((y_num - y_true)**2))
    return e_max, rmse

# 3. Численные методы
def solve_fde(alpha, y0, t_end, h):
    n = int(t_end / h)
    t = np.linspace(0, t_end, n + 1)
    
    y_l = np.zeros(n + 1); y_l[0] = y0
    y_m = np.zeros(n + 1); y_m[0] = y0
    y_t = np.zeros(n + 1); y_t[0] = y0
    
    coeff = 1 / gamma(alpha)
    
    for i in range(n):
        tn_next = t[i+1]
        
        # Левые прямоугольники
        s_l = 0
        for j in range(i + 1):
            w = ((tn_next - t[j])**alpha - (tn_next - t[j+1])**alpha) / alpha
            s_l += (y_l[j] + t[j]) * w
        y_l[i+1] = y0 + coeff * s_l

        # Средние прямоугольники
        s_m = 0
        for j in range(i + 1):
            t_mid = (t[j] + t[j+1]) / 2
            w = ((tn_next - t[j])**alpha - (tn_next - t[j+1])**alpha) / alpha
            s_m += (y_m[j] + t_mid) * w
        y_m[i+1] = y0 + coeff * s_m

        # Трапеции (Предиктор-Корректор)
        s_t = 0
        for j in range(i):
            w = ((tn_next - t[j])**alpha - (tn_next - t[j+1])**alpha) / alpha
            s_t += ((y_t[j] + t[j]) + (y_t[j+1] + t[j+1])) / 2 * w
        y_pred = y_t[i] + (h**alpha / gamma(alpha+1)) * (y_t[i] + t[i])
        w_last = ((tn_next - t[i])**alpha - (tn_next - t[i+1])**alpha) / alpha
        s_t += ((y_t[i] + t[i]) + (y_pred + tn_next)) / 2 * w_last
        y_t[i+1] = y0 + coeff * s_t
        
    return t, y_l, y_m, y_t

# --- Параметры из задания ---
alpha_val = 1.8
y0_val = 1.0
T = 20  # Оценка на длинном интервале
steps = [0.2, 0.1, 0.05, 0.025] # Шаги из таблицы

print(f"{'Метод':<20} | {'h':<6} | {'e_max':<12} | {'RMSE':<12}")
print("-" * 60)

for h in steps:
    t, yl, ym, yt = solve_fde(alpha_val, y0_val, T, h)
    y_true = analytical_solution(t, alpha_val, y0_val)
    
    for name, y_res in zip(["Левые", "Средние", "Трапеции"], [yl, ym, yt]):
        em, rm = calculate_metrics(y_res, y_true)
        print(f"{name:<20} | {h:<6} | {em:<12.5e} | {rm:<12.5e}")
    print("-" * 60)

# Построение графика накопления ошибки для последнего шага (T=20)
plt.figure(figsize=(10, 5))
plt.plot(t, np.abs(yl - y_true), label='Левые (ошибка)')
plt.plot(t, np.abs(ym - y_true), label='Средние (ошибка)')
plt.plot(t, np.abs(yt - y_true), label='Трапеции (ошибка)')
plt.yscale('log')
plt.title(f'Накопление ошибки на длинном интервале (T={T})')
plt.xlabel('t')
plt.ylabel('Абсолютная ошибка')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()