import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import brentq

def mittag_leffler(alpha, beta, z, tol=1e-15, max_terms=2000):
    """Вычисление функции Миттаг-Леффлера с двумя параметрами E_{alpha, beta}(z)."""
    result = 0.0
    for k in range(max_terms):
        term = z**k / gamma(alpha * k + beta)
        result += term
        if abs(term) < tol and k > int(5 + 2/alpha): 
            break
    return result

def analytical_solution(t, alpha, y0):
    """Точное решение уравнения D^alpha y(t) = y(t) + t."""
    y_true = np.zeros_like(t)
    for i, val in enumerate(t):
        if val == 0:
            y_true[i] = y0
            continue
        term1 = y0 * mittag_leffler(alpha, 1, val**alpha)
        term2 = (val**(alpha + 1)) * mittag_leffler(alpha, alpha + 2, val**alpha)
        y_true[i] = term1 + term2
    return y_true

def generate_custom_grid(alpha, t_end, h, T=None, p=2, delta=1e-12):
    """
    Генерация адаптированной сетки H_{T,h} с конца к началу
    на основе выравнивания локальной погрешности интегратора.
    """
    if T is None:
        T = t_end
        
    L = (h**(p + 1)) * (T**(alpha - p))
    t_nodes = [T]
    t_i = T
    
    while t_i > delta:
        def eq(x):
            x_safe = max(x, 1e-50)
            return ((t_i - x_safe)**(p + 1)) * (x_safe**(alpha - p)) - L
            
        try:
            if alpha - p < 0:
                left_bound = 1e-15
            else:
                left_bound = t_i * (alpha - p) / (alpha + 1)
                
            x_root = brentq(eq, left_bound, t_i - 1e-15)
        except ValueError:
            x_root = t_i - h
            
        if x_root <= delta:
            break
            
        t_nodes.append(x_root)
        t_i = x_root
        
    t_nodes.append(0.0) 
    t_nodes.reverse()   
    
    if T < t_end:
        current = T
        while current < t_end - 1e-9:
            current += h
            if current > t_end: current = t_end
            t_nodes.append(current)
            
    return np.array(t_nodes)


def solve_fde_left_rectangles(alpha, y0, t_end, h):
    """Метод левых прямоугольников (Явный метод Эйлера 1-го порядка)."""
    t = np.arange(0, t_end + h/2, h) 
    n = len(t) - 1
    y = np.zeros(n + 1)
    y[0] = y0
    coeff = 1 / gamma(alpha)
    
    for i in range(n):
        s = 0
        T = t[i+1]
        for j in range(i + 1):
            w_P = ((T - t[j])**alpha - (T - t[j+1])**alpha) / alpha
            s += w_P * (y[j] + t[j])
        y[i+1] = y[0] + coeff * s
    return y, t

def solve_fde_rk2_lubich(alpha, y0, t_end, h):
    """Метод Рунге-Кутты 2-го порядка (метод Хойна) по Любиху."""
    t = generate_custom_grid(alpha, t_end, h, T=t_end, p=2)
    print(f"Генерация сетки для метода РК2 (p=2)...", len(t))
    n = len(t) - 1 
    y = np.zeros(n + 1)
    y[0] = y0
    
    p_degree = alpha - 1
    coeff = 1 / gamma(p_degree)
    
    def compute_z(m, y_vals, y_pred=None):
        if m == 0: return 0.0
        T = t[m]
        z_val = 0.0
        for j in range(m):
            dt = t[j+1] - t[j]
            u_j = T - t[j]
            u_jp1 = T - t[j+1]
            
            term1 = (u_j**(p_degree+1) - u_jp1**(p_degree+1)) / (p_degree + 1)
            term2 = (u_j**p_degree - u_jp1**p_degree) / p_degree
            w_L = (term1 - u_jp1 * term2) / dt
            w_R = (u_j * term2 - term1) / dt
            
            f_j = y_vals[j] + t[j]
            f_jp1 = (y_pred + t[j+1]) if (j == m - 1 and y_pred is not None) else (y_vals[j+1] + t[j+1])
            z_val += w_L * f_j + w_R * f_jp1
        return coeff * z_val

    for i in range(n):
        h_i = t[i+1] - t[i]
        k1 = compute_z(i, y)
        Y_pred = y[i] + h_i * k1  
        k2 = compute_z(i+1, y, y_pred=Y_pred)
        y[i+1] = y[i] + (h_i / 2) * (k1 + k2)  
    return y, t

def solve_fde_pece(alpha, y0, t_end, h):
    """Дробный метод Адамса-Башфорта-Мултона (PECE)."""
    t = np.arange(0, t_end + h/2, h)
    print(f"Генерация сетки для метода PECE (p=2)...", len(t))
    n = len(t) - 1
    y = np.zeros(n + 1)
    y[0] = y0
    coeff = 1 / gamma(alpha)
    
    for i in range(n):
        T = t[i+1]
        
        # 1. ПРЕДИКТОР
        s_pred = 0
        for j in range(i + 1):
            w_P = ((T - t[j])**alpha - (T - t[j+1])**alpha) / alpha
            s_pred += w_P * (y[j] + t[j])
        y_pred = y[0] + coeff * s_pred
        
        # 2. КОРРЕКТОР
        s_corr = 0
        for j in range(i + 1):
            dt = t[j+1] - t[j]
            u_j = T - t[j]
            u_jp1 = T - t[j+1]
            
            term1 = (u_j**(alpha+1) - u_jp1**(alpha+1)) / (alpha + 1)
            term2 = (u_j**alpha - u_jp1**alpha) / alpha
            w_L = (term1 - u_jp1 * term2) / dt
            w_R = (u_j * term2 - term1) / dt
            
            f_j = y[j] + t[j]
            f_jp1 = (y_pred + t[i+1]) if j == i else (y[j+1] + t[j+1])
            s_corr += w_L * f_j + w_R * f_jp1
            
        y[i+1] = y[0] + coeff * s_corr
    return y, t



def calculate_metrics(y_num, y_true):
    e_max = np.max(np.abs(y_num - y_true))
    rmse = np.sqrt(np.mean((y_num - y_true)**2))
    return e_max, rmse

if __name__ == "__main__":
    alpha = 1.9
    y0 = 0.0      
    t_max = 1.0   
    h = 0.05

    # Вычисления
    y_l, t_val1 = solve_fde_left_rectangles(alpha, y0, t_max, h)
    y_rk2, t_val2 = solve_fde_rk2_lubich(alpha, y0, t_max, h)
    y_pece, t_val3 = solve_fde_pece(alpha, y0, t_max, h)
    
    y_real1 = analytical_solution(t_val1, alpha, y0)
    y_real2 = analytical_solution(t_val2, alpha, y0)
    y_real3 = analytical_solution(t_val3, alpha, y0)
    
    # Сбор метрик
    metrics = {
        "Левые прямоуг. (равномерная)": calculate_metrics(y_l, y_real1),
        "РК2 (VIDE, адаптивная)": calculate_metrics(y_rk2, y_real2),
        "PECE (Адамс, равномерная)": calculate_metrics(y_pece, y_real3),
    }

    print(f"{'Метод':<30} | {'e_max':<12} | {'RMSE':<12}")
    print("-" * 60)
    for method, (emax, rmse) in metrics.items():
        print(f"{method:<30} | {emax:<12.5e} | {rmse:<12.5e}")

    # Построение графиков
    plt.figure(figsize=(14, 6))

    # 1. Решения
    plt.subplot(1, 2, 1)
    plt.plot(t_val2, y_real2, 'k--', label='Аналитическое', linewidth=2)
    plt.plot(t_val1, y_l, 'r-', label='Явный Эйлер (равномерная)', alpha=0.8)
    plt.plot(t_val2, y_rk2, 'c-', label='РК2 (адаптивная сетка)', alpha=0.8)
    plt.plot(t_val3, y_pece, 'm-', label='PECE (равномерная)', alpha=0.8)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('y(t)', fontsize=12)
    plt.title(f'Решение уравнения $D^{{{alpha}}}y = y + t$',  fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. Погрешности
    plt.subplot(1, 2, 2)
    plt.plot(t_val1, np.abs(y_l - y_real1), 'r-', label='Ошибка Явного Эйлера')
    plt.plot(t_val2, np.abs(y_rk2 - y_real2), 'c-', label='Ошибка РК2 (адаптивная)')
    plt.plot(t_val3, np.abs(y_pece - y_real3), 'm-', label='Ошибка PECE', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('t', fontsize=12)
    plt.ylabel('Абсолютная ошибка |y_num - y_exact|', fontsize=12)
    plt.title('Анализ погрешности методов', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    
    # === ГЕНЕРАЦИЯ СПЕЦИАЛЬНОГО ГРАФИКА ДЛЯ ГЛАВЫ 3 ===
    print("\nГенерация сводного графика для альфа = 1.1 и 1.9...")
    plt.figure(figsize=(14, 6))
    
    alphas_to_plot = [1.1, 1.9]
    h_test = 0.05
    num_test = int((t_max - 0) / h_test) + 1
    t_unif = np.linspace(0, t_max, num_test)
    
    for i, alpha_val in enumerate(alphas_to_plot):
        # Вычисления для конкретного альфа
        y_l_a, t_val1_a = solve_fde_left_rectangles(alpha_val, y0, t_max, h_test)
        y_rk2_a, t_val2_a = solve_fde_rk2_lubich(alpha_val, y0, t_max, h_test)
        y_pece_a, t_val3_a = solve_fde_pece(alpha_val, y0, t_max, h_test)
        
        y_real1_a = analytical_solution(t_val1_a, alpha_val, y0)
        y_real2_a = analytical_solution(t_val2_a, alpha_val, y0)
        y_real3_a = analytical_solution(t_val3_a, alpha_val, y0)
        
        # Строим график ошибки в соответствующей панели (слева или справа)
        plt.subplot(1, 2, i + 1)
        plt.plot(t_val1_a, np.abs(y_l_a - y_real1_a), 'r-', label='Ошибка Явного Эйлера')
        plt.plot(t_val2_a, np.abs(y_rk2_a - y_real2_a), 'c-', label='Ошибка РК2 (адаптивная)')
        plt.plot(t_val3_a, np.abs(y_pece_a - y_real3_a), 'm-', label='Ошибка PECE', linewidth=2)
        
        plt.yscale('log')
        plt.xlabel('t', fontsize=12)
        plt.ylabel('Абсолютная ошибка |y_num - y_exact|', fontsize=12)
        plt.title(f'Анализ погрешности при $\\alpha={alpha_val}$', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    # Сохраняем картинку прямо в папку со скриптом, чтобы сразу вставить в диплом!
    plt.savefig('Figure_3_alpha_comparison.png', dpi=300)
    plt.show()