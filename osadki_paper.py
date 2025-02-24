import math
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
temp_osadki = pd.read_csv("temp_osad.txt", sep=";", names=['kod', 'year', 'month', 'day', 'prizn', 'T_min', 'T_sr', 'T_max', 'H_mm'])
osadki_poln = temp_osadki[['year', 'month', 'day', 'H_mm']]
osadki_poln['H_mm'] = pd.to_numeric(osadki_poln['H_mm'], errors='coerce')
osadki = osadki_poln.dropna()
osadki = osadki[osadki['H_mm'] != 0]

# Сортировка данных
osadki_sort = osadki.sort_values(by=['H_mm'], ascending=False)
osadki_sort['date'] = pd.to_datetime(osadki_sort[['year', 'month', 'day']], format='%Y-%m-%d')
osadki_sort = osadki_sort.drop(['year', 'month', 'day'], axis=1)

# Удаление дубликатов
osadki_no_dup = osadki_sort.drop_duplicates(subset=['H_mm']).reset_index(drop=True)
osadki_no_dup['numer'] = osadki_no_dup.index + 1
osadki_no_dup['P'] = osadki_no_dup['numer'] / (len(osadki_no_dup) + 1)
osadki_no_dup['P%'] = osadki_no_dup['P'] * 100

# Оценка параметров распределения Пирсона III
try:
    a1, loc1, scale1 = sts.pearson3.fit(osadki_no_dup['H_mm'])
    print(a1, loc1, scale1)
except Exception as e:
    print(f"Error fitting Pearson III distribution: {e}")
    raise

# Теоретическая функция распределения
pear = sts.pearson3(a1, loc1, scale1)
f1_x = pear.cdf(osadki_no_dup['H_mm'])

# Преобразование вероятностей для вероятностной сетки
osadki_no_dup['F_inv'] = sts.norm.ppf(1 - f1_x)

# Создание графика с вероятностной сеткой
plt.figure(figsize=(8, 5))

# Неравномерная шкала для оси Y (логарифмическая)
plt.yscale('log')  # Логарифмическая шкала для оси Y

# Настройка меток на оси Y
y_ticks = np.logspace(np.log10(osadki_no_dup['H_mm'].min()), np.log10(osadki_no_dup['H_mm'].max()), num=10)
plt.yticks(y_ticks, [f"{y:.1f}" for y in y_ticks])

# Эмпирические точки
plt.plot(osadki_no_dup['F_inv'], osadki_no_dup['H_mm'], '+', label='Эмпирические точки')

# Теоретическая прямая
x_theory = np.linspace(min(osadki_no_dup['F_inv']), max(osadki_no_dup['F_inv']), 100)
y_theory = sts.pearson3.ppf(1 - sts.norm.cdf(x_theory), a1, loc1, scale1)
plt.plot(x_theory, y_theory, label='Теоретическая кривая')

# Настройка графика
plt.xlabel("Нормированная вероятность")
plt.ylabel("Суточные осадки, мм (логарифмическая шкала)")
plt.title("Вероятностная сетка для распределения Пирсона III")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()

# Отображение графика
plt.show()

# Рассчет обеспеченных высот осадков
rjad_P = [0.5, 1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 92, 98, 99]
H_obesp = [[i, round(sts.pearson3.isf(i / 100, a1, loc1, scale1))] for i in rjad_P]
obesp_osadki = pd.DataFrame(H_obesp, columns=['P%', 'Hmm']).set_index('P%')

# Вывод таблицы в формате LaTeX
print(obesp_osadki.to_latex(float_format="%.1f"))