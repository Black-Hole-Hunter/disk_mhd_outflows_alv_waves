
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# настройка размера шрифта на рисунке
fsize=12
plt.rcParams.update({'axes.titlesize':fsize})
plt.rcParams.update({'axes.labelsize':fsize})
plt.rcParams.update({'font.size':fsize})
plt.rcParams.update({'xtick.labelsize':fsize})
plt.rcParams.update({'ytick.labelsize':fsize})
plt.rcParams.update({'legend.fontsize':fsize-4})

# относительный путь к файлам данных
path = "./out6/"

# список файлов для визуализации
files_list = [0, 100, 500, 700, 800, 831]

# 1 дюйм в сантиметрах
inch2cm = 2.54
# ширина рисунка в см
fig_width_cm = 20  # Эта строка была добавлена
# высота рисунка в см
fig_height_cm = 10 # Эта строка была добавлена
# размер маркера
ms = 0.5

# создание окна рисунка с 1-ой панелью 
fig, axs = plt.subplots(1, 1, figsize=(fig_width_cm/inch2cm, fig_height_cm/inch2cm), dpi=200)

axs.set_xlabel(r'$z/H$')
axs.set_ylabel(r'$v/v_0, v_A/v_0$')

r_unit = 1
v_unit = 1

for file in files_list:

        f = open(path + "data" + str(file) + ".dat")
        t = f.readline()
        print("t=", t)
        data1 = np.loadtxt(path + "data" + str(file) + ".dat", skiprows=1)

        axs.plot(data1[:, 0]/r_unit, data1[:, 3], 'o-', markersize=ms, linewidth=1, label=f' t= {float(t):.1e}')
        axs.plot(data1[:, 0]/r_unit, data1[:, 6], 'x-', markersize=ms, linewidth=1, label=f'al,t= {float(t):.1e}')

axs.legend(loc='best')
plt.tight_layout()
plt.show()
