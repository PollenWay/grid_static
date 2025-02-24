import math
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
temp_osadki = pd.read_csv("temp_osad.txt", sep=";", names=['kod', 'year', 'month', 'day', 'prizn', 'T_min', 'T_sr', 'T_max', 'H_mm'])
temp_osadki['T_min'] = pd.to_numeric(temp_osadki['T_min'], errors='coerce')

# Фильтрация данных: отрицательные значения T_min в зимний период
T_min = temp_osadki[['year', 'month', 'day', 'T_min']].dropna()
T_min = T_min[T_min['T_min'] < 0]
T_min = T_min[(T_min['month'] <= 3) | (T_min['month'] >= 11)]

# Удаление дубликатов и сортировка
T_min_no_dup = T_min.drop_duplicates(subset=['T_min']).reset_index(drop=True)
T_min_sort = T_min_no_dup.sort_values(by=['T_min'], ascending=False).reset_index(drop=True)

# Добавление столбцов для вероятностей
T_min_sort['date'] = pd.to_datetime(T_min_sort[['year', 'month', 'day']])
T_min_sort = T_min_sort.drop(['year', 'month', 'day'], axis=1)
T_min_sort = T_min_sort.reindex(columns=['date', 'T_min'])
T_min_sort['numer'] = T_min_sort.index + 1
T_min_sort['P'] = T_min_sort['numer'] / (len(T_min_sort) + 1)
T_min_sort['P%'] = T_min_sort['P'] * 100

# Сохранение данных в файл
T_min_sort.to_csv('min_zim.txt', index=False)
print(T_min_sort.to_latex(float_format="%.1f"))

# Оценка параметров распределения Пирсона III
a1, loc1, scale1 = sts.pearson3.fit(T_min_sort['T_min'])
print(f"Параметры распределения Пирсона III: a1={a1}, loc1={loc1}, scale1={scale1}")

# Теоретическая функция распределения
pear = sts.pearson3(a1, loc1, scale1)
f1_x = pear.cdf(T_min_sort['T_min'])

# Преобразование вероятностей для вероятностной сетки
T_min_sort['F_inv'] = sts.norm.ppf(1 - f1_x)

# Генерация теоретической прямой на основе тех же F_inv значений
y_theory = sts.pearson3.ppf(1 - sts.norm.cdf(T_min_sort['F_inv']), a1, loc1, scale1)

# Построение графика вероятностной сетки
plt.figure(figsize=(8, 5))

# Неравномерная шкала для оси Y (линейная для температуры)
plt.yticks(np.arange(-50, 0, 5))

# Настройка меток на оси X (вероятностная шкала)
x_ticks = np.arange(0, 105, 5)
x_tick_labels = [f"{sts.norm.cdf(sts.norm.ppf((100 - x) / 100)):.2f}" for x in x_ticks]
plt.xticks(x_ticks, labels=x_tick_labels, rotation=90)

# Эмпирические точки
plt.plot(T_min_sort['P%'], T_min_sort['T_min'], '+', label='Эмпирические точки')

# Теоретическая прямая (используем те же значения P% для y_theory)
plt.plot(T_min_sort['P%'], y_theory, label='Теоретическая кривая')

# Настройка графика
plt.xlabel("Вероятность % (нормированная вероятностная шкала)")
plt.ylabel("Температура, °C")
plt.title("Вероятностная сетка для распределения Пирсона III")
plt.legend()
plt.grid(True)
plt.show()

# Рассчет обеспеченных температур
rjad_P = [0.5, 1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 92, 95, 97, 98, 99]
T_obesp = [[i, round(sts.pearson3.isf(i / 100, a1, loc1, scale1))] for i in rjad_P]

# Создание DataFrame для обеспеченностей
obesp_T = pd.DataFrame(T_obesp, columns=['P%', 'T_min']).set_index('P%')
print(obesp_T.to_latex(float_format="%.1f"))