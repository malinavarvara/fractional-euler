import numpy as np
import matplotlib.pyplot as plt

# Импортируем ваши функции (убедитесь, что файл project.py в той же папке)
from qwerty import solve_fde_left_rectangles, solve_fde_rk2_lubich, solve_fde_pece, analytical_solution, calculate_metrics

def plot_log_log_analysis():
    alpha = 1.8
    y0 = 0.0      
    t_max = 1.0   
    # Набор шагов h для анализа (рефаймент)
    h_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    
    # Списки для хранения погрешностей
    errors_euler = []
    errors_rk2 = []
    errors_pece = []

    print(f"Запуск h-рефаймента для alpha = {alpha}...")

    for h in h_values:
        # 1. Расчеты
        y_l, t_l = solve_fde_left_rectangles(alpha, y0, t_max, h)
        y_rk2, t_rk2 = solve_fde_rk2_lubich(alpha, y0, t_max, h)
        y_pece, t_pece = solve_fde_pece(alpha, y0, t_max, h)
        
        # 2. Точные решения для каждой сетки
        y_true_l = analytical_solution(t_l, alpha, y0)
        y_true_rk2 = analytical_solution(t_rk2, alpha, y0)
        y_true_pece = analytical_solution(t_pece, alpha, y0)
        
        # 3. Сбор e_max
        errors_euler.append(calculate_metrics(y_l, y_true_l)[0])
        errors_rk2.append(calculate_metrics(y_rk2, y_true_rk2)[0])
        errors_pece.append(calculate_metrics(y_pece, y_true_pece)[0])

    # Настройка стиля для диплома (Times New Roman, если установлен)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 12,
        "grid.alpha": 0.5
    })

    plt.figure(figsize=(8, 7))

    # Построение основных линий
    plt.loglog(h_values, errors_euler, 'ro-', label='Левые прямоугольники ($O(h^1)$)', markersize=8)
    plt.loglog(h_values, errors_rk2, 'cs-', label='РК2 VIDE ($O(h^2)$)', markersize=8)
    plt.loglog(h_values, errors_pece, 'm^-', label='PECE Адамс ($O(h^2)$)', markersize=8, linewidth=2)

    # Отрисовка эталонных треугольников наклона (для наглядности)
    # Для Эйлера (наклон 1)
    h_ref = np.array([h_values[0], h_values[1]])
    plt.loglog(h_ref, errors_euler[0] * (h_ref/h_ref[0])**1, 'k--', alpha=0.5)
    plt.text(h_ref[1], errors_euler[1]*1.2, 'наклон 1', fontsize=10)

    # Для PECE (наклон 2)
    plt.loglog(h_ref, errors_pece[0] * (h_ref/h_ref[0])**2, 'k--', alpha=0.5)
    plt.text(h_ref[1], errors_pece[0]*(h_ref[1]/h_ref[0])**2 * 0.5, 'наклон 2', fontsize=10)

    plt.gca().invert_xaxis() # Шаг уменьшается слева направо
    plt.xlabel('Шаг сетки $h$', fontsize=13)
    plt.ylabel('Максимальная ошибка $e_{max}$', fontsize=13)
    plt.title(f'Зависимость погрешности от шага $h$ (Log-Log)\nПри $\\alpha = {alpha}$', fontsize=14)
    plt.grid(True, which="both", linestyle="--")
    plt.legend(fontsize=11, loc='lower left')

    plt.tight_layout()
    plt.savefig('log_log_error.png', dpi=300) # Сохраняем для вставки в LaTeX
    plt.show()

if __name__ == "__main__":
    plot_log_log_analysis()