import math
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
snow = pd.read_csv("snow_kosh.txt", sep=";", names=['kod', 'year', 'month', 'day', 'hsnow'])
del snow['kod']
snow = snow.loc[snow['hsnow'] != 9999]
snow = snow.loc[snow['hsnow'] != 0]

# Сортировка данных
snow_sort = snow.sort_values(by=['hsnow'], ascending=False, inplace=False)
snow_sort = snow_sort.reset_index()
del snow_sort['index']

# Расчет вероятностей
snow_sort['numer'] = snow_sort.index + 1
snow_sort['P'] = snow_sort['numer'] / (len(snow_sort) + 1)
snow_sort['P%'] = snow_sort['P'] * 100

# Оценка параметров распределения Пирсона III
a1, loc1, scale1 = sts.pearson3.fit(snow_sort['hsnow'])

# Теоретическая функция распределения
f1_x = sts.pearson3.cdf(snow_sort['hsnow'], a1, loc1, scale1)

# Преобразование вероятностей для вероятностной сетки
snow_sort['F_inv'] = sts.norm.ppf(1 - f1_x)

# Градации для оси X
gradaciy = math.ceil(5 * math.log10(len(snow_sort)))
delta = round((snow_sort['hsnow'].max() - snow_sort['hsnow'].min()) / gradaciy, 1)

# Создание вероятностной сетки
plt.figure(figsize=(8, 5))

# Неравномерная шкала для оси Y (логарифмическая)
plt.yscale('log')  # Логарифмическая шкала для оси Y

# Настройка меток на оси Y
y_ticks = np.logspace(math.log10(snow_sort['hsnow'].min()), math.log10(snow_sort['hsnow'].max()), num=10)  # 10 равномерных значений в логарифмическом масштабе
plt.yticks(y_ticks, [f"{y:.1f}" for y in y_ticks])  # Добавляем подписи к меткам

# Эмпирические точки
plt.plot(snow_sort['F_inv'], snow_sort['hsnow'], '+', label='Эмпирические точки')

# Теоретическая прямая
x_theory = np.linspace(min(snow_sort['F_inv']), max(snow_sort['F_inv']), 100)
y_theory = sts.pearson3.ppf(1 - sts.norm.cdf(x_theory), a1, loc1, scale1)
plt.plot(x_theory, y_theory, label='Теоретическая кривая')

# Настройка графика
plt.xlabel("Нормированная вероятность")
plt.ylabel("Высота снега, см (логарифмическая шкала)")
plt.title("Вероятностная сетка для распределения Пирсона III")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)  # Включаем сетку для обеих шкал
plt.legend()

# Отображение графика
plt.show()

# Рассчет обеспеченных высот снега
H_obesp = {}
for i in [0.5, 1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 97, 99]:
    P = sts.pearson3.isf(i / 100, a1, loc1, scale1)
    H_obesp.update({str(i): P})

# Вывод результатов в DataFrame
teor = pd.DataFrame(list(H_obesp.keys()), list(H_obesp.values()))
teor_transpose = teor.T
print(teor_transpose.to_latex(float_format="%.1f"))