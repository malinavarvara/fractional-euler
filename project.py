import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 1. Функция Миттаг-Леффлера для аналитического решения E_{alpha, beta}(z)
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

# Точное решение уравнения D^alpha y = y + t
def analytical_solution(t, alpha, y0):
    y_true = np.zeros_like(t)
    for i, val in enumerate(t):
        # Формула: y(t) = y0*E_{a,1}(t^a) + t^a*E_{a, a+1}(t^a)
        term1 = y0 * mittag_leffler(alpha, 1, val**alpha)
        term2 = (val**alpha) * mittag_leffler(alpha, alpha + 1, val**alpha)
        y_true[i] = term1 + term2
    return y_true

# 2. Численные методы решения дробного уравнения
def solve_fde_all_methods(alpha, y0, t_end, h):
    n = int(t_end / h)
    t = np.linspace(0, t_end, n + 1)
    
    y_left = np.zeros(n + 1); y_left[0] = y0
    y_mid = np.zeros(n + 1); y_mid[0] = y0
    y_trap = np.zeros(n + 1); y_trap[0] = y0
    
    coeff = 1 / gamma(alpha)
    
    for i in range(n):
        tn_next = t[i+1]
        
        # --- Метод левых прямоугольников (Явная схема) ---
        sum_l = 0
        for j in range(i + 1):
            weight = ((tn_next - t[j])**alpha - (tn_next - t[j+1])**alpha) / alpha
            sum_l += (y_left[j] + t[j]) * weight
        y_left[i+1] = y0 + coeff * sum_l

        # --- Метод средних прямоугольников ---
        sum_m = 0
        for j in range(i + 1):
            t_mid = (t[j] + t[j+1]) / 2
            y_mid_point = y_mid[j] # Упрощенный прогноз
            weight = ((tn_next - t[j])**alpha - (tn_next - t[j+1])**alpha) / alpha
            sum_m += (y_mid_point + t_mid) * weight
        y_mid[i+1] = y0 + coeff * sum_m

        # --- Метод Трапеций (Предиктор-Корректор) ---
        sum_t = 0
        for j in range(i):
            w = ((tn_next - t[j])**alpha - (tn_next - t[j+1])**alpha) / alpha
            f_avg = ((y_trap[j] + t[j]) + (y_trap[j+1] + t[j+1])) / 2
            sum_t += f_avg * w
        
        # Предиктор для последней точки
        y_pred = y_trap[i] + (h**alpha / gamma(alpha+1)) * (y_trap[i] + t[i])
        f_avg_last = ((y_trap[i] + t[i]) + (y_pred + tn_next)) / 2
        w_last = ((tn_next - t[i])**alpha - (tn_next - t[i+1])**alpha) / alpha
        sum_t += f_avg_last * w_last
        
        y_trap[i+1] = y0 + coeff * sum_t
        
    return t, y_left, y_mid, y_trap

# 3. Расчет метрик погрешности
def calculate_metrics(y_num, y_true):
    # Максимальная абсолютная погрешность e_max
    e_max = np.max(np.abs(y_num - y_true))
    # Среднеквадратическая погрешность RMSE
    rmse = np.sqrt(np.mean((y_num - y_true)**2))
    return e_max, rmse

# --- Основной блок выполнения ---
if __name__ == "__main__":
    # Параметры задачи
    alpha = 1.8   # Порядок производной
    y0 = 1.0      # Начальное условие
    t_max = 12.0   # Конечный момент времени
    h = 0.025       # Шаг сетки

    # Вычисления
    t, y_l, y_m, y_t = solve_fde_all_methods(alpha, y0, t_max, h)
    y_real = analytical_solution(t, alpha, y0)

    # Расчет ошибок
    metrics = {
        "Левые прямоугольники": calculate_metrics(y_l, y_real),
        "Средние прямоугольники": calculate_metrics(y_m, y_real),
        "Трапеции": calculate_metrics(y_t, y_real)
    }

    # Вывод в консоль
    print(f"{'Метод':<25} | {'e_max':<12} | {'RMSE':<12}")
    print("-" * 55)
    for method, (emax, rmse) in metrics.items():
        print(f"{method:<25} | {emax:<12.5e} | {rmse:<12.5e}")

    # Визуализация
    plt.figure(figsize=(15, 6))

    # График 1: Решения
    plt.subplot(1, 2, 1)
    plt.plot(t, y_real, 'k--', label='Аналитическое ($E_{\\alpha, \\beta}$)', linewidth=2.5)
    plt.plot(t, y_l, 'r-', label='Левые прямоуг.', alpha=0.7)
    plt.plot(t, y_m, 'g-', label='Средние прямоуг.', alpha=0.7)
    plt.plot(t, y_t, 'b-', label='Трапеции', alpha=0.7)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title(f'Решение уравнения $D^{{{alpha}}}y = y + t$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # График 2: Погрешности
    plt.subplot(1, 2, 2)
    plt.plot(t, np.abs(y_l - y_real), 'r-', label='Ошибка левых')
    plt.plot(t, np.abs(y_m - y_real), 'g-', label='Ошибка средних')
    plt.plot(t, np.abs(y_t - y_real), 'b-', label='Ошибка трапеций')
    plt.yscale('log') # Логарифмическая шкала для лучшей видимости разницы
    plt.xlabel('t')
    plt.ylabel('Абсолютная ошибка')
    plt.title('Погрешность методов (логарифмическая шкала)')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()