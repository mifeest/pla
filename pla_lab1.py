import numpy as np

# Алфавит и кодирование
alp = list('абвгдежзийклмнопрстуфхцчшщъыьэюя')
alp_cod = {}
for i in range(len(alp)):
    str_bin = format(i, '05b')  
    alp_cod[alp[i]] = str_bin

word = 'стоп'
bin_list = [alp_cod[ch] for ch in word]
bin_word = ''.join(bin_list)

# Формируем матрицу данных: 4 строки (буквы), 5 столбцов (биты)
data_bits = np.zeros((4, 5), dtype=int)
for i, bit in enumerate(bin_word):
    data_bits[i // 5, i % 5] = int(bit)

# Генераторная матрица G (4×7) для (7,4)-кода Хэмминга
G = np.array([
    [1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 1]
], dtype=int)

# Матрица проверки H (3×7)
H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1]
], dtype=int)

# КОДИРОВАНИЕ
# Применяем код к каждому столбцу (битовой позиции)
# Результат: encoded — матрица 7×5 (7 закодированных битов для каждой из 5 позиций)
encoded = np.zeros((7, 5), dtype=int)
for col in range(5):
    info_bits = data_bits[:, col]  # вектор длины 4
    code_word = (info_bits @ G) % 2  # вектор длины 7
    encoded[:, col] = code_word

# ВНЕСЕНИЕ ОШИБОК
p = encoded.copy()
p[0, 1] = 0  # инверсия (если был 1 → 0, и наоборот)
p[3, 2] = 1
p[2, 3] = 0
p[5, 4] = 1

# ДЕКОДИРОВАНИЕ И ИСПРАВЛЕНИЕ
# Вычисляем синдром для каждого столбца
syndrome = (H @ p) % 2  # shape (3, 5)

# Подготовим обратное отображение: синдром → позиция ошибки (0–6)
syndrome_to_pos = {}
for pos in range(7):
    col = tuple(H[:, pos])  # столбец H как кортеж
    syndrome_to_pos[col] = pos

# Исправляем ошибки
corrected = p.copy()
for col in range(5):
    s = tuple(syndrome[:, col])
    if s != (0, 0, 0):
        if s in syndrome_to_pos:
            error_pos = syndrome_to_pos[s]
            corrected[error_pos, col] ^= 1  # инвертируем бит

# ВОССТАНОВЛЕНИЕ ИСХОДНЫХ ДАННЫХ
# Берём первые 4 строки (информационные биты) из исправленной матрицы
recovered_data = corrected[:4, :]  # shape (4, 5)

# Преобразуем обратно в строку битов
recovered_bits = []
for row in range(4):
    for col in range(5):
        recovered_bits.append(str(recovered_data[row, col]))
bin_end = ''.join(recovered_bits)

# Разбиваем на 5-битные блоки и декодируем в буквы
reversed_alp_code = {v: k for k, v in alp_cod.items()}
word_end_list = []
for i in range(0, 20, 5):
    block = bin_end[i:i+5]
    word_end_list.append(reversed_alp_code[block])

word_end = ''.join(word_end_list)

print("Восстановленное слово:", word_end)

