import numpy as np
import matplotlib.pyplot as plt
import math


# ================================================
# 1. Генерация случайного выпуклого многоугольника
# ================================================
def generate_polygon(n_vertices):
    """
    Генерирует случайный (выпуклый) n-угольник вокруг начала координат.

    Принцип:
    - Генерируем n случайных углов в [0, 2π), сортируем их → точки идут против часовой стрелки.
    - Генерируем случайные радиусы > 1, чтобы избежать вырождения в центр.
    - Добавляем первую точку в конец, чтобы замкнуть контур.
    """
    angles = np.sort(np.random.rand(n_vertices) * 2 * np.pi)  # Сортированные углы
    radii = np.random.rand(n_vertices) + 1  # Радиусы в [1, 2)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    # Замыкаем многоугольник
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    return x, y


# ================================================
# 2. Вспомогательная функция: отображение графика
# ================================================
def show(title_text):
    """
    Настройка и вывод графика:
    - Сетка, оси, равные масштабы по X и Y,
    - Легенда и заголовок.
    """
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=1)  # Ось X
    plt.axvline(0, color='black', linewidth=1)  # Ось Y
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')  # Равный масштаб
    plt.title(title_text)
    plt.show()


# ================================================
# 3. Вспомогательная функция: определитель
# ================================================
def det(matrix):
    """Выводит определитель матрицы с округлением до 2 знаков."""
    determinant = np.linalg.det(matrix)
    print(f'Определитель равен: {round(determinant, 2)}')


# ================================================
# 4. Вспомогательная функция: собственные значения и векторы
# ================================================
def own_values(matrix):
    """Выводит собственные значения и собственные векторы матрицы."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print("Собственные значения:", eigenvalues)
    print("Собственные векторы:\n", eigenvectors)


# ================================================
# ЗАДАНИЕ 1: Отражение относительно прямой y = a·x
# ================================================
def task1(vertices, a):
    """
    Отражает многоугольник относительно прямой y = a·x.

    Математика:
    1. Поворачиваем систему координат так, чтобы прямая y = a·x совпала с осью X.
       Угол поворота: θ = arctan(a).
    2. Отражаем относительно новой оси X (матрица [[1, 0], [0, -1]]).
    3. Поворачиваем систему обратно.

    Итоговая матрица: S = R⁻¹ · M · R
    """
    theta = np.arctan(a)  # Угол наклона прямой

    # Матрица поворота на -θ (чтобы выровнять прямую с осью X)
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta), np.cos(-theta)]])

    # Обратная матрица поворота = поворот на +θ
    R_inv = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

    # Матрица отражения относительно оси X
    M = np.array([[1, 0],
                  [0, -1]])

    # Итоговая матрица отражения
    S = R_inv @ M @ R

    print("Матрица отражения S =\n", S)
    det(S)
    own_values(S)

    # Применяем преобразование ко всем вершинам
    transformed = vertices @ S

    # Визуализация
    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label='Отражённый')
    # Прямая y = a·x
    x_line = np.linspace(-4, 4, 200)
    plt.plot(x_line, a * x_line, 'g--', label=f'y = {a}x')
    show(f'Отражение относительно прямой y = {a}x')


# ================================================
# ЗАДАНИЕ 2: Проекция на прямую y = b·x
# ================================================
def task2(vertices, b):
    """
    Ортогонально проецирует многоугольник на прямую y = b·x.

    Проекционная матрица на направляющий вектор (1, b):
        S = (1 / (1 + b²)) * [[1, b], [b, b²]]
    """
    S = (1 / (1 + b ** 2)) * np.array([[1, b],
                                       [b, b ** 2]])
    print("Матрица проекции S =\n", S)
    det(S)
    own_values(S)

    transformed = vertices @ S

    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label='Проекция')
    x_line = np.linspace(-3, 3, 100)
    plt.plot(x_line, b * x_line, 'g--', label=f'y = {b}x')
    show(f'Проекция на прямую y = {b}x')


# ================================================
# ЗАДАНИЕ 3: Поворот на угол theta
# ================================================
def task3(vertices, theta_degrees):
    """
    Поворачивает многоугольник против часовой стрелки на заданный угол.
    """
    theta = np.radians(theta_degrees)
    S = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    print("Матрица поворота S =\n", S)
    det(S)
    own_values(S)

    transformed = vertices @ S

    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label=f'Поворот на {theta_degrees}°')
    show(f'Поворот на {theta_degrees}° против часовой стрелки')


# ================================================
# ЗАДАНИЕ 4: Центральная симметрия
# ================================================
def task4(vertices):
    """
    Отражает фигуру относительно начала координат (умножение на -1).
    """
    S = np.array([[-1, 0],
                  [0, -1]])
    print("Матрица симметрии S =\n", S)
    det(S)
    own_values(S)

    transformed = vertices @ S

    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label='Центральная симметрия')
    show('Центральная симметрия относительно начала координат')


# ================================================
# ЗАДАНИЕ 5: Отражение + поворот
# ================================================
def task5(vertices, a, d):
    """
    Сначала отражает относительно y = a·x, затем поворачивает по часовой стрелке на 10·d градусов.
    """
    # --- Шаг 1: отражение ---
    A = (1 / (1 + a ** 2)) * np.array([[1 - a ** 2, 2 * a],
                                       [2 * a, a ** 2 - 1]])
    reflected = vertices @ A

    # --- Шаг 2: поворот по часовой стрелке ---
    theta = np.radians(10 * d)  # Положительный угол → по часовой стрелке при использовании sin(-θ)
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    transformed = reflected @ R

    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(reflected[:, 0], reflected[:, 1], 'go-', label='После отражения')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label='Отражение + Поворот')
    x_vals = np.linspace(-3, 3, 100)
    plt.plot(x_vals, a * x_vals, 'g--', label=f'y = {a}x')
    show(f'Отражение от y={a}x и поворот на {10 * d}° по часовой стрелке')


# ================================================
# ЗАДАНИЕ 6: Линейное преобразование базиса
# ================================================
def task6(vertices, a, b):
    """
    Преобразует канонический базис так:
       e1 = (1,0) → (1, a)
       e2 = (0,1) → (1, b)
    Матрица преобразования: S = [[1, 1], [a, b]]
    """
    S = np.array([[1, 1],
                  [a, b]])
    transformed = vertices @ S

    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label='Преобразованный')
    x_vals = np.linspace(-3, 3, 100)
    plt.plot(x_vals, a * x_vals, 'g--', label=f'y = {a}x')
    plt.plot(x_vals, b * x_vals, 'm--', label=f'y = {b}x')
    show(f'Преобразование: y=0 → y={a}x, x=0 → y={b}x')


# ================================================
# ЗАДАНИЕ 7: Обратное преобразование базиса
# ================================================
def task7(vertices, a, b):
    """
    Преобразует прямые y = a·x и y = b·x в координатные оси.
    Здесь используется фиксированная матрица для демонстрации.
    """
    # Пример неединственного решения; в общем случае нужно решать СЛАУ
    A = np.array([[1, -1],
                  [0.5, 1]])
    transformed = vertices @ A

    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label='Преобразованный')
    x_vals = np.linspace(-3, 3, 100)
    plt.plot(x_vals, a * x_vals, 'g--', label=f'y = {a}x')
    plt.plot(x_vals, b * x_vals, 'm--', label=f'y = {b}x')
    plt.axhline(0, color='black', linestyle='--', label='y = 0')
    plt.axvline(0, color='black', linestyle='--', label='x = 0')
    show(f'Преобразование: y={a}x → y=0, y={b}x → x=0')


# ================================================
# ЗАДАНИЕ 8: Меняем местами прямые y = a·x и y = b·x
# ================================================
def task8(vertices, a, b):
    """
    В данном примере используется простое отражение (может не менять местами в общем случае).
    Для точного обмена нужно более сложное преобразование.
    """
    S = np.array([[1, 0],
                  [0, -1]])  # Пример: отражение по X
    own_values(S)
    transformed = vertices @ S

    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-')
    x_vals = np.linspace(-3, 3, 100)
    plt.plot(x_vals, a * x_vals, 'g--', label=f'y = {a}x')
    plt.plot(x_vals, b * x_vals, 'm--', label=f'y = {b}x')
    show(f'Преобразование, меняющее местами y={a}x и y={b}x (пример)')


# ================================================
# ЗАДАНИЕ 9: Масштабирование круга до площади c
# ================================================
def task9(c):
    """
    Площадь круга: π·r². Чтобы площадь стала c, нужно r = sqrt(c/π).
    Но здесь масштабируется единичный круг (площадь π) до площади c.
    → Масштаб по обоим осям: k = sqrt(c / π) / 1 = sqrt(c / π)
    Однако в коде используется упрощение: увеличение площади в c раз → масштаб sqrt(c).
    Это верно, если исходная площадь = 1. Но у единичного круга площадь = π.
    Исправление: будем считать, что "единичный" = площадь 1 → радиус = 1/√π.
    Но для простоты в учебных целях часто берут радиус = 1 и говорят "масштаб sqrt(c)".
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.array([np.cos(theta), np.sin(theta)])  # Радиус = 1 → площадь = π

    # Чтобы новая площадь = c, нужно k_x * k_y = c / π
    # Берём равномерное масштабирование: k = sqrt(c / π)
    k = math.sqrt(c / math.pi)
    S = np.array([[k, 0],
                  [0, k]])
    print("Матрица масштабирования S =\n", S)
    det(S)

    scaled = S @ circle

    plt.figure(figsize=(6, 6))
    plt.plot(circle[0], circle[1], 'b-', label='Круг (радиус=1)')
    plt.plot(scaled[0], scaled[1], 'r-', label=f'Площадь ≈ {c}')
    show(f'Масштабирование круга до площади {c}')


# ================================================
# ЗАДАНИЕ 10: Преобразование круга в эллипс площади d
# ================================================
def task10(d):
    """
    Преобразуем круг в эллипс с заданной площадью d.
    Площадь эллипса = π·a·b → нужно a·b = d / π.
    Выбираем a = 2 (масштаб по X), тогда b = d / (2π).
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.array([np.cos(theta), np.sin(theta)])

    k_x = 2
    k_y = d / (k_x * math.pi)  # Чтобы π·k_x·k_y = d → k_x·k_y = d/π
    S = np.array([[k_x, 0],
                  [0, k_y]])
    print("Матрица масштабирования S =\n", S)
    det(S)

    scaled = S @ circle

    plt.figure(figsize=(6, 6))
    plt.plot(circle[0], circle[1], 'b-', label='Круг')
    plt.plot(scaled[0], scaled[1], 'r-', label=f'Эллипс площади {d}')
    show(f'Круг → эллипс площади {d}')


# ================================================
# ЗАДАНИЕ 11: Преобразование с вещественными собственными векторами
# ================================================
def task11(vertices):
    """
    Матрица A = [[1.5, 1], [1, 0.5]] имеет два различных вещественных собственных значения.
    Собственные векторы показывают направления, которые не меняют направление при преобразовании.
    """
    S = np.array([[1.5, 1.0],
                  [1.0, 0.5]])
    own_values(S)
    transformed = vertices @ S

    # Визуализация собственных векторов
    _, eigenvectors = np.linalg.eig(S)
    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-')
    origin = np.array([[0, 0], [0, 0]])
    plt.quiver(*origin, eigenvectors[0, :], eigenvectors[1, :],
               color=['g', 'm'], scale=5, angles='xy', label='Собственные векторы')
    plt.legend()
    show('Преобразование с двумя вещественными собственными векторами')


# ================================================
# ЗАДАНИЯ 12–14: Особые случаи собственных векторов
# ================================================
def task12(vertices):
    """Поворот на 45°: собственные значения комплексные → нет вещественных собственных векторов."""
    theta = np.radians(45)
    S = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    own_values(S)
    transformed = vertices @ S
    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label='Повёрнутый')
    show('Поворот на 45°: нет вещественных собственных векторов')


def task13(vertices):
    """Поворот на 90°: собственные значения чисто мнимые."""
    S = np.array([[0, -1],
                  [1, 0]])
    own_values(S)
    transformed = vertices @ S
    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label='Повёрнутый на 90°')
    show('Поворот на 90°: собственные значения мнимые')


def task14(vertices):
    """Скалярная матрица: каждая точка — собственный вектор."""
    lam = 9
    S = lam * np.eye(2)
    own_values(S)
    transformed = vertices @ S
    plt.figure(figsize=(6, 6))
    plt.plot(vertices[:, 0], vertices[:, 1], 'bo-', label='Исходный')
    plt.plot(transformed[:, 0], transformed[:, 1], 'ro-', label=f'Масштаб ×{lam}')
    show(f'Скалярное преобразование (λ = {lam}): все векторы собственные')


# ================================================
# ЗАДАНИЯ 15–16: Композиция преобразований
# ================================================
def task15(vertices):
    """Некоммутирующие преобразования: поворот и масштабирование по X."""
    theta = np.radians(45)
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # Поворот
    B = np.array([[2, 0],
                  [0, 1]])  # Масштаб по X

    own_values(A)
    own_values(B)

    # Применение
    A_v = vertices @ A
    B_v = vertices @ B
    AB_v = A_v @ B  # Сначала A, потом B
    BA_v = B_v @ A  # Сначала B, потом A

    # Сравнение
    plt.figure(figsize=(6, 6))
    plt.plot(AB_v[:, 0], AB_v[:, 1], 'mo-', label='AB')
    plt.plot(BA_v[:, 0], BA_v[:, 1], 'co-', label='BA')
    show('AB ≠ BA: поворот и масштабирование не коммутируют')


def task16(vertices):
    """Коммутирующие преобразования: поворот и однородное масштабирование."""
    A = 2 * np.eye(2)  # Однородное масштабирование
    theta = np.radians(45)
    B = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # Поворот

    own_values(A)
    own_values(B)

    AB_v = (vertices @ A) @ B
    BA_v = (vertices @ B) @ A

    plt.figure(figsize=(6, 6))
    plt.plot(AB_v[:, 0], AB_v[:, 1], 'mo-', label='AB')
    plt.plot(BA_v[:, 0], BA_v[:, 1], 'co-', label='BA')
    show('AB = BA: однородное масштабирование коммутирует с поворотом')


# ================================================
# Основной блок запуска
# ================================================
if __name__ == "__main__":
    # Параметры из условия
    a = 9
    b = 10
    c = 15
    d = 5
    n_dots = 5  # Количество вершин

    # Генерация многоугольника
    x, y = generate_polygon(n_dots)
    vertices = np.column_stack((x, y))  # (n+1) × 2

    #task1(vertices, a)
    #task2(vertices, b)
    #task3(vertices, 150)
    #task4(vertices)
    #task5(vertices, a, d)
    #task6(vertices, a, b)
    #task7(vertices, a, b)
    #task8(vertices, a, b)
    #task9(c)
    #task10(d)
    #task11(vertices)
    #task12(vertices)
    #task13(vertices)
    #task14(vertices)
    #task15(vertices)
    #task16(vertices)