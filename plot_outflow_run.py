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
path = "./out/"
#"./out/"
# список файлов для визуализации
files_list = [0,100,500,700,800,831]
#files_list = [0, 100]


# 1 дюйм в сантиметрах
inch2cm = 2.54
# ширина рисунка в см
fig_width_cm = 20
# высота рисунка в см
fig_height_cm = 10
# размер маркера
ms = 0.5

# создание окна рисунка с 2-мя панелями
fig, axs = plt.subplots(2, 2, sharex=True, figsize=(fig_width_cm/inch2cm, fig_height_cm/inch2cm), dpi=200)

axs[0][0].set_ylabel(r'$\rho/\rho_0$')
# axs[0][0].grid()
axs[0][0].set_yscale("log")

axs[0][1].set_ylabel(r'$v_\varphi/v_0$')
# axs[0][1].grid()

axs[1][0].set_xlabel(r'$z/H$')
# axs[1][0].set_ylabel(r'$L/L_0$')
axs[1][0].set_ylabel(r'$v/v_0$')
# axs[1][0].grid()

axs[1][1].set_xlabel(r'$z/H$')
axs[1][1].set_ylabel(r'$B/B_0$')
# axs[1][1].grid()

# единица измерения координат
r_unit = 1 
# единица измерения скорости
v_unit = 1 

for file in files_list:
    f = open(path + "data" + str(file) + ".dat")
    t = f.readline()
    print("t=", t)

    data1 = np.loadtxt(path + "data" + str(file) + ".dat", skiprows=1)   
    axs[0][0].plot(data1[:,0]/r_unit, data1[:,1], 'o-', markersize=ms, fillstyle='none', linewidth=1, label=("t= %.1e" % float(t)))
    axs[0][1].plot(data1[:,0]/r_unit, data1[:,2], 'o-', markersize=ms, fillstyle='none', linewidth=1)
    axs[1][0].plot(data1[:,0]/r_unit, data1[:,3], 'o-', markersize=ms, fillstyle='none', linewidth=1)
    axs[1][1].plot(data1[:,0]/r_unit, data1[:,4], 'o-', markersize=ms, fillstyle='none', linewidth=1)
    axs[1][0].plot(data1[:, 0]/r_unit, data1[:, 6], 'x-', markersize=ms, linewidth=1, label=r'$v_A/v_0$')

axs[0][0].legend(loc= 'best')
plt.tight_layout()

plt.show()
# fig.savefig("run_1.png")