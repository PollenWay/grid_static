import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QAction, QVBoxLayout, QWidget, QLabel, QFrame
from PyQt5.QtGui import QFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import math
import scipy.stats as sts

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Метеорологические данные")
        self.setGeometry(100, 100, 1000, 500)

        # Создаем основной виджет и layout
        self.main_frame = QWidget()
        self.layout = QVBoxLayout()
        self.main_frame.setLayout(self.layout)
        self.setCentralWidget(self.main_frame)

        # Шрифт
        self.custom_font = QFont("DejaVu Sans", 12)

        # Меню
        self.menu_bar = self.menuBar()

        # Вкладка "Параметры"
        self.params_menu = self.menu_bar.addMenu("Параметры")

        # Вкладка "Сетка"
        self.grid_menu = self.menu_bar.addMenu("Сетка")

        # Добавляем пункты меню для "Параметры"
        self.params_actions = {}
        for param in ["Температура", "Роза ветров", "Осадки", "Снежный покров", "Облачность"]:
            action = QAction(param, self)
            action.triggered.connect(lambda checked, p=param: self.on_params_select(p))
            self.params_menu.addAction(action)
            self.params_actions[param] = action

        # Добавляем пункты меню для "Сетка"
        self.grid_actions = {}
        for grid_type in ["Температура", "Осадки", "Снежный покров"]:
            action = QAction(grid_type, self)
            action.triggered.connect(lambda checked, g=grid_type: self.on_grid_select(g))
            self.grid_menu.addAction(action)
            self.grid_actions[grid_type] = action

        # Метка для выбранного параметра или сетки
        self.selected_label = QLabel("Выберите параметр или тип сетки", self)
        self.selected_label.setFont(self.custom_font)
        self.layout.addWidget(self.selected_label)

        # Фрейм для графика
        self.graph_frame = QFrame()
        self.layout.addWidget(self.graph_frame)

    def on_params_select(self, selected_value):
        """Обработчик выбора элемента из меню 'Параметры'."""
        self.selected_label.setText(f"Вы выбрали: {selected_value}")
        self.clear_graph_frame()

        if selected_value == "Облачность":
            self.process_cloudiness_data()
        elif selected_value == "Роза ветров":
            self.process_wind_rose_data()
        elif selected_value == "Осадки":
            self.process_precipitation_data()
        elif selected_value == "Снежный покров":
            self.process_snow_data()
        elif selected_value == "Температура":
            self.process_temperature_data()

    def on_grid_select(self, selected_value):
        """Обработчик выбора элемента из меню 'Сетка'."""
        self.selected_label.setText(f"Вы выбрали: {selected_value} (вероятностная сетка)")
        self.clear_graph_frame()

        if selected_value == "Температура":
            self.plot_temperature_grid()
        elif selected_value == "Осадки":
            self.plot_precipitation_grid()
        elif selected_value == "Снежный покров":
            self.plot_snow_grid()

    def clear_graph_frame(self):
        """Очистка фрейма графика."""
        for widget in self.graph_frame.children():
            if isinstance(widget, FigureCanvas):
                widget.deleteLater()

    def process_cloudiness_data(self):
        """Обработка данных по облачности и отрисовка графика."""
        data = pd.read_csv('obl.txt', sep=";",
                           names=['kod', 'year', 'cloud_type', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                                  'Sep', 'Oct', 'Nov', 'Dec'])

        # Фильтрация данных
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            data = data.loc[data[month] != 9999]
            data = data.loc[data[month] != 0]

        cloudiness_long = data.melt(
            id_vars=['kod', 'year', 'cloud_type'],
            value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            var_name='month',
            value_name='cloudiness'
        )

        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        cloudiness_long['month'] = pd.Categorical(cloudiness_long['month'], categories=month_order, ordered=True)

        monthly_stats = cloudiness_long.groupby('month', observed=True)['cloudiness'].agg(
            mean='mean',
            median='median',
            std='std',
            min='min',
            max='max'
        ).reset_index()

        # Создаем график
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(monthly_stats['month'], monthly_stats['mean'], color='skyblue', label='Среднее значение')
        ax.errorbar(
            monthly_stats['month'],
            monthly_stats['mean'],
            yerr=monthly_stats['std'],
            fmt='o',
            color='red',
            label='Стандартное отклонение'
        )
        ax.set_xlabel("Месяц", fontsize=10)
        ax.set_ylabel("Средняя облачность, баллы", fontsize=10)
        ax.set_title("Среднемесячная облачность по месяцам", fontsize=12)
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvas(fig)
        self.graph_frame.layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.layout.addWidget(canvas)

    def process_wind_rose_data(self):
        """Обработка данных для построения розы ветров."""
        wind_data = pd.read_csv("wind_rose.txt", sep=";",
                                names=['kod', 'year', 'month', 'day', 'hour', 'rumb', 'V_sr', 'V_max'])

        wind_data['rumb'] = pd.to_numeric(wind_data['rumb'], errors='coerce')
        wind_data = wind_data.dropna(subset=['rumb'])
        wind_data = wind_data[wind_data['rumb'] != 0]

        ugly = [360, 315, 270, 225, 180, 135, 90, 45]
        nacalo = 0

        def get_wind_directions(data):
            nord = data.loc[(data['rumb'] >= 338) | (data['rumb'] <= 22)]
            nord_ost = data.loc[(data['rumb'] >= 23) & (data['rumb'] <= 67)]
            ost = data.loc[(data['rumb'] >= 68) & (data['rumb'] <= 112)]
            south_ost = data.loc[(data['rumb'] >= 113) & (data['rumb'] <= 157)]
            south = data.loc[(data['rumb'] >= 158) & (data['rumb'] <= 202)]
            south_west = data.loc[(data['rumb'] >= 203) & (data['rumb'] <= 247)]
            west = data.loc[(data['rumb'] >= 248) & (data['rumb'] <= 292)]
            nord_west = data.loc[(data['rumb'] >= 293) & (data['rumb'] <= 337)]

            napr = [
                nord['V_sr'].count(), nord_ost['V_sr'].count(), ost['V_sr'].count(),
                south_ost['V_sr'].count(), south['V_sr'].count(), south_west['V_sr'].count(),
                west['V_sr'].count(), nord_west['V_sr'].count()
            ]

            total_count = sum(napr)
            napr_proc = [(i / total_count) * 100 for i in napr]

            rad = [math.radians(i) for i in ugly]
            X = [math.sin(rad[i]) * napr_proc[i] + nacalo for i in range(8)]
            Y = [math.cos(rad[i]) * napr_proc[i] + nacalo for i in range(8)]
            X_L = [[nacalo, X[i]] for i in range(8)]
            Y_L = [[nacalo, Y[i]] for i in range(8)]

            return X, Y, X_L, Y_L, napr_proc

        f1 = get_wind_directions(wind_data.loc[(wind_data['month'] >= 4) & (wind_data['month'] <= 10)])  # Тёплый период
        f2 = get_wind_directions(wind_data)  # Годовая роза ветров
        f3 = get_wind_directions(wind_data.loc[(wind_data['month'] <= 4) | (wind_data['month'] >= 10)])  # Холодный период

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        def plot_wind_rose(ax, data, title):
            ax.plot(data[0], data[1], 'ro')  # Красные маркеры для точек
            ax.plot(data[0], data[1], '-', lw=3)  # Линия между точками
            for i in range(8):
                ax.plot([data[2][i][0], data[2][i][1]], [data[3][i][0], data[3][i][1]], 'b')  # Линии к центру
                ax.text(data[0][i], data[1][i], str(round(data[4][i], 1)), backgroundcolor='w')  # Текстовые метки
            ax.set_title(title)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        plot_wind_rose(axes[0], f2, 'Годовая роза ветров')
        plot_wind_rose(axes[1], f1, 'Роза ветров за тёплый период')
        plot_wind_rose(axes[2], f3, 'Роза ветров за холодный период')

        plt.tight_layout()

        canvas = FigureCanvas(fig)
        self.graph_frame.layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.layout.addWidget(canvas)

    def process_temperature_data(self):
        """Обработка данных по температуре и отрисовка графика."""
        temp_osadki = pd.read_csv("temp_osad.txt", sep=";",
                                  names=['kod', 'year', 'month', 'day', 'prizn', 'T_min', 'T_sr', 'T_max', 'H_mm'])

        temp_osadki['T_min'] = pd.to_numeric(temp_osadki['T_min'], errors='coerce')
        T_min = temp_osadki[['year', 'month', 'day', 'T_min']].dropna()
        T_min = T_min[T_min['T_min'] < 0]
        T_min = T_min[(T_min['month'] <= 3) | (T_min['month'] >= 11)]

        T_min_no_dup = T_min.drop_duplicates(subset=['T_min']).reset_index(drop=True)
        T_min_sort = T_min_no_dup.sort_values(by=['T_min'], ascending=False).reset_index(drop=True)

        T_min_sort['date'] = pd.to_datetime(T_min_sort[['year', 'month', 'day']])
        T_min_sort = T_min_sort.drop(['year', 'month', 'day'], axis=1)
        T_min_sort['numer'] = T_min_sort.index + 1
        T_min_sort['P%'] = (T_min_sort['numer'] / (len(T_min_sort) + 1)) * 100

        try:
            a1, loc1, scale1 = sts.pearson3.fit(T_min_sort['T_min'])
            print(f"Параметры распределения Пирсона III: a={a1}, loc={loc1}, scale={scale1}")
        except Exception as e:
            print(f"Ошибка при подборе распределения Пирсона III: {e}")
            raise

        pear = sts.pearson3(a1, loc1, scale1)
        f1_x = pear.cdf(T_min_sort['T_min'])
        T_min_sort['F_inv'] = sts.norm.ppf(1 - f1_x)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(T_min_sort['P%'], T_min_sort['T_min'], '+', label='Эмпирические точки')
        ax.plot(T_min_sort['P%'], sts.pearson3.ppf(1 - sts.norm.cdf(T_min_sort['F_inv']), a1, loc1, scale1), label='Теоретическая кривая')
        ax.set_xlabel("Вероятность %", fontsize=10)
        ax.set_ylabel("Температура, °C", fontsize=10)
        ax.set_xticks(np.arange(0.0, 105, 5.0))
        ax.set_yticks(np.arange(-50, 0, 5))
        ax.legend()
        ax.grid(True)
        ax.set_title("Распределение минимальной температуры", fontsize=12)

        canvas = FigureCanvas(fig)
        self.graph_frame.layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.layout.addWidget(canvas)

    def process_precipitation_data(self):
        """Обработка данных по осадкам и отрисовка графика."""
        temp_osadki = pd.read_csv("temp_osad.txt", sep=";",
                                  names=['kod', 'year', 'month', 'day', 'prizn', 'T_min', 'T_sr', 'T_max', 'H_mm'])

        osadki_poln = temp_osadki[['year', 'month', 'day', 'H_mm']].copy()
        osadki_poln.loc[:, 'H_mm'] = pd.to_numeric(osadki_poln['H_mm'], errors='coerce')
        osadki = osadki_poln.dropna(subset=['H_mm'])
        osadki = osadki[osadki['H_mm'] != 0]

        osadki_sort = osadki.sort_values(by=['H_mm'], ascending=False)
        osadki_no_dup = osadki_sort.drop_duplicates(subset=['H_mm']).reset_index(drop=True)
        osadki_no_dup['numer'] = osadki_no_dup.index + 1
        osadki_no_dup['P%'] = osadki_no_dup['numer'] / (len(osadki_no_dup) + 1) * 100

        try:
            a1, loc1, scale1 = sts.pearson3.fit(osadki_no_dup['H_mm'])
            print(f"Параметры распределения Пирсона III: a={a1}, loc={loc1}, scale={scale1}")
        except Exception as e:
            print(f"Ошибка при подборе распределения Пирсона III: {e}")
            raise

        pear = sts.pearson3(a1, loc1, scale1)
        f1_x = pear.cdf(osadki_no_dup['H_mm'])
        osadki_no_dup['F_inv'] = sts.norm.ppf(1 - f1_x)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(osadki_no_dup['P%'], osadki_no_dup['H_mm'], '+', label='Эмпирические точки')
        ax.plot(osadki_no_dup['P%'], sts.pearson3.ppf(1 - sts.norm.cdf(osadki_no_dup['F_inv']), a1, loc1, scale1), label='Теоретическая кривая')
        ax.set_xlabel("Вероятность %", fontsize=10)
        ax.set_ylabel("Суточные осадки, мм", fontsize=10)
        ax.set_xticks(np.arange(0.0, 105, 5.0))
        ax.set_yticks(np.arange(0, 80, 5))
        ax.legend()
        ax.grid(True)
        ax.set_title("Распределение суточных осадков", fontsize=12)

        canvas = FigureCanvas(fig)
        self.graph_frame.layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.layout.addWidget(canvas)

    def process_snow_data(self):
        """Обработка данных по снежному покрову и отрисовка графика."""
        snow = pd.read_csv("snow_kosh.txt", sep=";", names=['kod', 'year', 'month', 'day', 'hsnow'])
        del snow['kod']

        snow = snow.loc[snow['hsnow'] != 9999]
        snow = snow.loc[snow['hsnow'] != 0]
        snow_sort = snow.sort_values(by=['hsnow'], ascending=False, inplace=False)
        snow_sort = snow_sort.reset_index(drop=True)
        snow_sort['numer'] = snow_sort.index + 1
        snow_sort['P'] = snow_sort['numer'] / (len(snow_sort) + 1)
        snow_sort['P%'] = snow_sort['P'] * 100

        try:
            a1, loc1, scale1 = sts.pearson3.fit(snow_sort['hsnow'])
            print(f"Параметры распределения Пирсона III: a={a1}, loc={loc1}, scale={scale1}")
        except Exception as e:
            print(f"Ошибка при подборе распределения Пирсона III: {e}")
            raise

        pear = sts.pearson3(a1, loc1, scale1)
        f1_x = pear.cdf(snow_sort['hsnow'])
        snow_sort['F_inv'] = sts.norm.ppf(1 - f1_x)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(snow_sort['P%'], snow_sort['hsnow'], '+', label='Эмпирические точки')
        ax.plot(snow_sort['P%'], sts.pearson3.ppf(1 - sts.norm.cdf(snow_sort['F_inv']), a1, loc1, scale1), label='Теоретическая кривая')
        ax.set_xlabel("Вероятность %", fontsize=10)
        ax.set_ylabel("Высота снега, см", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 50)
        ax.set_xticks(np.arange(0.0, 105, 5.0))
        ax.set_yticks(np.arange(math.trunc(min(snow_sort['hsnow'])), 50, 5))
        ax.legend()
        ax.grid(True)
        ax.set_title("Распределение высоты снежного покрова", fontsize=12)

        canvas = FigureCanvas(fig)
        self.graph_frame.layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.layout.addWidget(canvas)

    def plot_temperature_grid(self):
        """Построение вероятностной сетки для температуры."""
        temp_osadki = pd.read_csv("temp_osad.txt", sep=";",
                                  names=['kod', 'year', 'month', 'day', 'prizn', 'T_min', 'T_sr', 'T_max', 'H_mm'])

        temp_osadki['T_min'] = pd.to_numeric(temp_osadki['T_min'], errors='coerce')
        T_min = temp_osadki[['year', 'month', 'day', 'T_min']].dropna()
        T_min = T_min[T_min['T_min'] < 0]
        T_min = T_min[(T_min['month'] <= 3) | (T_min['month'] >= 11)]

        T_min_no_dup = T_min.drop_duplicates(subset=['T_min']).reset_index(drop=True)
        T_min_sort = T_min_no_dup.sort_values(by=['T_min'], ascending=False).reset_index(drop=True)
        T_min_sort['numer'] = T_min_sort.index + 1
        T_min_sort['P'] = T_min_sort['numer'] / (len(T_min_sort) + 1)
        T_min_sort['P%'] = T_min_sort['P'] * 100

        try:
            a1, loc1, scale1 = sts.pearson3.fit(T_min_sort['T_min'])
            print(f"Параметры распределения Пирсона III: a={a1}, loc={loc1}, scale={scale1}")
        except Exception as e:
            print(f"Ошибка при подборе распределения Пирсона III: {e}")
            raise

        pear = sts.pearson3(a1, loc1, scale1)
        f1_x = pear.cdf(T_min_sort['T_min'])
        T_min_sort['F_inv'] = sts.norm.ppf(1 - f1_x)

        x_theory = np.linspace(min(T_min_sort['F_inv']), max(T_min_sort['F_inv']), 100)
        y_theory = sts.pearson3.ppf(1 - sts.norm.cdf(x_theory), a1, loc1, scale1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(T_min_sort['F_inv'], T_min_sort['T_min'], '+', label='Эмпирические точки')
        ax.plot(x_theory, y_theory, label='Теоретическая кривая')
        ax.set_xlabel("Нормированная вероятность", fontsize=10)
        ax.set_ylabel("Температура, °C", fontsize=10)
        ax.set_yticks(np.arange(-50, 0, 5))
        ax.legend()
        ax.grid(True)
        ax.set_title("Вероятностная сетка для температуры", fontsize=12)

        canvas = FigureCanvas(fig)
        self.graph_frame.layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.layout.addWidget(canvas)

    def plot_precipitation_grid(self):
        """Построение вероятностной сетки для осадков."""
        temp_osadki = pd.read_csv("temp_osad.txt", sep=";",
                                  names=['kod', 'year', 'month', 'day', 'prizn', 'T_min', 'T_sr', 'T_max', 'H_mm'])

        osadki_poln = temp_osadki[['year', 'month', 'day', 'H_mm']].copy()
        osadki_poln.loc[:, 'H_mm'] = pd.to_numeric(osadki_poln['H_mm'], errors='coerce')
        osadki = osadki_poln.dropna(subset=['H_mm'])
        osadki = osadki[osadki['H_mm'] != 0]

        osadki_sort = osadki.sort_values(by=['H_mm'], ascending=False)
        osadki_no_dup = osadki_sort.drop_duplicates(subset=['H_mm']).reset_index(drop=True)
        osadki_no_dup['numer'] = osadki_no_dup.index + 1
        osadki_no_dup['P%'] = osadki_no_dup['numer'] / (len(osadki_no_dup) + 1) * 100

        try:
            a1, loc1, scale1 = sts.pearson3.fit(osadki_no_dup['H_mm'])
            print(f"Параметры распределения Пирсона III: a={a1}, loc={loc1}, scale={scale1}")
        except Exception as e:
            print(f"Ошибка при подборе распределения Пирсона III: {e}")
            raise

        pear = sts.pearson3(a1, loc1, scale1)
        f1_x = pear.cdf(osadki_no_dup['H_mm'])
        osadki_no_dup['F_inv'] = sts.norm.ppf(1 - f1_x)

        x_theory = np.linspace(min(osadki_no_dup['F_inv']), max(osadki_no_dup['F_inv']), 100)
        y_theory = sts.pearson3.ppf(1 - sts.norm.cdf(x_theory), a1, loc1, scale1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(osadki_no_dup['F_inv'], osadki_no_dup['H_mm'], '+', label='Эмпирические точки')
        ax.plot(x_theory, y_theory, label='Теоретическая кривая')
        ax.set_xlabel("Нормированная вероятность", fontsize=10)
        ax.set_ylabel("Суточные осадки, мм (логарифмическая шкала)", fontsize=10)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        ax.set_title("Вероятностная сетка для осадков", fontsize=12)

        canvas = FigureCanvas(fig)
        self.graph_frame.layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.layout.addWidget(canvas)

    def plot_snow_grid(self):
        """Построение вероятностной сетки для снежного покрова."""
        snow = pd.read_csv("snow_kosh.txt", sep=";", names=['kod', 'year', 'month', 'day', 'hsnow'])
        del snow['kod']

        snow = snow.loc[snow['hsnow'] != 9999]
        snow = snow.loc[snow['hsnow'] != 0]
        snow_sort = snow.sort_values(by=['hsnow'], ascending=False, inplace=False)
        snow_sort = snow_sort.reset_index(drop=True)
        snow_sort['numer'] = snow_sort.index + 1
        snow_sort['P'] = snow_sort['numer'] / (len(snow_sort) + 1)
        snow_sort['P%'] = snow_sort['P'] * 100

        try:
            a1, loc1, scale1 = sts.pearson3.fit(snow_sort['hsnow'])
            print(f"Параметры распределения Пирсона III: a={a1}, loc={loc1}, scale={scale1}")
        except Exception as e:
            print(f"Ошибка при подборе распределения Пирсона III: {e}")
            raise

        pear = sts.pearson3(a1, loc1, scale1)
        f1_x = pear.cdf(snow_sort['hsnow'])
        snow_sort['F_inv'] = sts.norm.ppf(1 - f1_x)

        x_theory = np.linspace(min(snow_sort['F_inv']), max(snow_sort['F_inv']), 100)
        y_theory = sts.pearson3.ppf(1 - sts.norm.cdf(x_theory), a1, loc1, scale1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(snow_sort['F_inv'], snow_sort['hsnow'], '+', label='Эмпирические точки')
        ax.plot(x_theory, y_theory, label='Теоретическая кривая')
        ax.set_xlabel("Нормированная вероятность", fontsize=10)
        ax.set_ylabel("Высота снега, см (логарифмическая шкала)", fontsize=10)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)
        ax.set_title("Вероятностная сетка для снежного покрова", fontsize=12)

        canvas = FigureCanvas(fig)
        self.graph_frame.layout = QVBoxLayout(self.graph_frame)
        self.graph_frame.layout.addWidget(canvas)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())