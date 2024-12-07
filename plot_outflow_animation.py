import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# настройка размера шрифта на рисунке
fsize=12
plt.rcParams.update({'axes.titlesize':fsize})
plt.rcParams.update({'axes.labelsize':fsize})
plt.rcParams.update({'font.size':fsize})
plt.rcParams.update({'xtick.labelsize':fsize})
plt.rcParams.update({'ytick.labelsize':fsize})
plt.rcParams.update({'legend.fontsize':fsize-2})

path = "./out/tmp/"
# path = "C:/Users/khaib/research/disk_wind_modeling_1d/data_WAS24/1/run1_eps=200/"

file0 = 0 # номер первого файла данных
fileN = 4500 # номер последнего файла данных
dfile = 500 # шаг по файлам - с каким интервалом их считывать
f_list = [] # список файлов для анимации
for i in range(int(fileN/dfile)):
    file_n = (i+1)*dfile
    f_list.append(file_n)
    
# f_list.append(783)
files_list = f_list


# 1 дюйм в сантиметрах
inch2cm = 2.54
# ширина рисунка в см
fig_width_cm = 20
# высота рисунка в см
fig_height_cm = 10

# создание окна рисунка с 2-мя панелями
fig, axs = plt.subplots(1, 1, sharex=True, figsize=(fig_width_cm/inch2cm, fig_height_cm/inch2cm), dpi=200)

axs.set_xlabel(r'$z/H$')
axs.set_ylabel(r'$v_z/v_0$')
axs.grid()
# axs.set_xlim(0, 5)
axs.axvline(x=5.0, color='black')
    
# единица измерения координат
r_unit = 1 
# единица измерения скорости
v_unit = 1 

ims=[]
for file in files_list:
    f = open(path + "data" + str(file) + ".dat")
    t = f.readline()
    print("t=", t)
    data1 = np.loadtxt(path + "data" + str(file) + ".dat", skiprows=1)   
    
    im = axs.plot(data1[:,0]/r_unit, data1[:,3], 'o-', color='orange', markersize=1.0, linewidth=1)
    
    ims.append(im)

# fig.legend()
plt.tight_layout()

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,                          repeat_delay=1000)
plt.show()
# ani.save("v(z)_eps=300_C=0.5_N=2048.gif", dpi=300)