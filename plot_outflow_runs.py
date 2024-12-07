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
plt.rcParams.update({'legend.fontsize':fsize-2})

path = ""
# files_list = [0, 10, 20]
files_list = [0, 100, 840]


# 1 дюйм в сантиметрах
inch2cm = 2.54
# ширина рисунка в см
fig_width_cm = 20
# высота рисунка в см
fig_height_cm = 10

# создание окна рисунка с 2-мя панелями
fig, axs = plt.subplots(2, 2, sharex=True, figsize=(fig_width_cm/inch2cm, fig_height_cm/inch2cm), dpi=200)

axs[0][0].set_ylabel(r'$\rho/\rho_0$')
axs[0][0].grid()
axs[0][0].set_yscale("log")

axs[0][1].set_ylabel(r'$v_\varphi/v_0$')
axs[0][1].grid()

axs[1][0].set_xlabel(r'$z/H$')
# axs[1][0].set_ylabel(r'$L/L_0$')
axs[1][0].set_ylabel(r'$v_z/v_0$')
axs[1][0].grid()

axs[1][1].set_xlabel(r'$z/H$')
axs[1][1].set_ylabel(r'$B/B_0$')
axs[1][1].grid()

for ax in axs.flat:
    ax.set_xlim(0, 5)
    ax.axvline(x=1.0, color='black')
    
# единица измерения координат
r_unit = 1 
# единица измерения скорости
v_unit = 1 

for file in files_list:
    data1 = np.loadtxt(path + "data" + str(file) + ".dat")   
    axs[0][0].plot(data1[:,0]/r_unit, data1[:,1], 'o-', markersize=1.0, linewidth=1, label=str(file))
    # axs[0][0].axvline(x=1.0, color='black')
    axs[0][1].plot(data1[:,0]/r_unit, data1[:,2], 'o-', markersize=1.0, linewidth=1, label=str(file))
    axs[1][0].plot(data1[:,0]/r_unit, data1[:,3], 'o-', markersize=1.0, linewidth=1, label=str(file))
    axs[1][1].plot(data1[:,0]/r_unit, data1[:,4], 'o-', markersize=1.0, linewidth=1, label=str(file))


# axs[0].legend(loc= 'upper left')
plt.tight_layout()

fig.show()