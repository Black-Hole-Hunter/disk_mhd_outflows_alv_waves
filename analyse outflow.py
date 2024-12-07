import numpy as np
import math 
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")

# ----- константы -------------------------------------------------------------
au = 1.5e13        # астрономическая единица, см
G = 6.67e-8        # универсальная грав. постоянная, СГС
Msun = 1.99e33      # масса Слнца, г
Mstar = 1.0 * Msun # масса звезды, в массах Солнца
Rg = 8.31e7        # универсальная газовая постоянная, СГС
mu = 2.3           # молекулярный вес газа
day = 3600 * 24    # 1 день, сек
year = 365*24*3600 # 1 год, сек
m_p = 1.67e-24     # масса протона, г

# ----- параметры схемы -------------------------------------------------------
N = 512   # число узлов
Ngs = 1   # яисло фиктивных ячеек с каждой стороны
Ntot = N + 2*Ngs # полное число ячеек
t = 0     # текущее время
c = 0.95  # Число Куранта

L = 10    # длина расчетной области, в единицах шкалы высот диска H
a = 0     # координата левой границы расчетной области
b = L     # координата правой границы расчетной области

eps = 200.0 # параметр искусственной вязкости

# ----- параметры физ. модели -------------------------------------------------
r_cm = 4.97 * au          # радиальное расстояние от звезды, а.е.
z_d = 5.0                  # полутолщина диска, в единицах шкалы высот диска H
n_ISM = 1e9                # концентрация газа в мезвездной среде, см^(-3)
rho_ISM = mu * m_p * n_ISM # плотность МЗС, г/см^3
beta = 29.0                # плазменный параметр в диске
Bz = 1.0                   # безразмерная компонента Bz

# 0 - расчет без вращения, 1 - с вращением
rotation_flag = 1.0 
# 0 - расчет без гравитации, 1 - с гравитацией
gravity_flag = 1.0

# ----- масштабы безразмерных переменных --------------------------------------
rho0 = 1.78e-11   # температура в экватор. плоскости диска, г/см^3
T0 = 60        # температура внутри диска, К
c_Td = np.sqrt(Rg * T0 / mu) # скорость звука внутри диска, см/с
v0 = c_Td
H = 1.76e-1*au     # шкала высот диска, см
t0 = H / v0     # шкала измерения времени, сек
p0 = rho0*v0**2 # шкала измерения давления, дин
B0 = np.sqrt(8.0*np.pi*p0/beta) # шкала измерения маг. поля, Гс
g0 = v0**2/H    # шкала измерения ускорения, см/с^2

# ----- коэффициенты уравнений ------------------------------------------------

Ta = 380                    # температура в атомосфере, К
c_Ta = np.sqrt(Rg * Ta / mu) # скорость звука в атмосфере, см/с

# кеплеровская скорость, см/с
def v_k(r_au):
    return np.sqrt(G * Mstar / (r_au * au))

# кеплеровская скорость в начальной точке, см/с
vk0 = v_k(r_cm / au)

# скорость вращения в случае центробежного равновесия, см/с
def v_phi0(r_cm, z_cm):
    return v_k(r_cm / au) * np.power(1.0 + (z_cm / r_cm)**2, -0.75)

# профиль плотности изотермического гидростатического диска, беразмерная
# здесь z - безразмерная координата (в единицах H)
def rho_disk(z):
    return np.exp(-0.5 * (vk0/c_Td)**2 * (1.0 - (1.0 + (z*H/r_cm)**2.0)**(-0.5)))

rho_surf = rho_disk(z_d)*T0/Ta
# профиль плотности изотермической гидростатической атмосферы диска, беразмерная
# здесь z - безразмерная координата (в единицах H)
def rho_atm(z):
    # плотность на текущей высоте
    rho_at_z = rho_surf * np.exp((vk0/c_Ta)**2 * ((1.0 + (z*H/r_cm)**2.0)**(-0.5) - (1.0 + (z_d*H/r_cm)**2.0)**(-0.5)))
    if rho_at_z > (rho_ISM / rho0):
        return rho_surf * np.exp((vk0/c_Ta)**2 * ((1.0 + (z*H/r_cm)**2.0)**(-0.5) - (1.0 + (z_d*H/r_cm)**2.0)**(-0.5)))
    else:
        return rho_ISM / rho0

# ускорение силы тяжести звезды, см/с^2
def g_z(r_cm, z_cm, M_g):
    return gravity_flag * G*M_g*z_cm/np.power(r_cm**2 + z_cm**2, 1.5)

def va(rho_atm):
    try:
        return B0/np.sqrt(4*np.pi*rho_atm)
    except ZeroDivisionError:
        return 0  # Или другое значение по умолчанию при нулевой плотности

# альвеновская скорость в экватор. плоскости диска в начале расчета, см/с
v_A_disk = B0 / np.sqrt(4.0 * np.pi * rho0)
# альвеновская скорость на верхней гранцие расчетной области, в начале расчета, см/с
v_A_L = B0 / np.sqrt(4.0 * np.pi * rho_atm(L))
# безразмерный квадрат максимальной скорости звука на сетке
cTcT = max((c_Td/v0)**2, (c_Ta/v0)**2)
# время окончания расчета (безразмерное)
t_stop = 0.2
# начальная максимальная скорость на сетке - скорость газа плюс магнитозвуковая скорость в области максимальных T и B, см/с
v_max = vk0 + np.sqrt(max(v_A_disk, v_A_L)**2 + max(c_Td, c_Ta)**2)
# Число Маха в диске
Mach = vk0 / c_Td

print("Model scales:")
print(" rho0   = %.3e [g/cm^3]" % (rho0))
print(" t0     = %.2e [yr] = %.2e [d]" % (t0/year, t0/day))
print(" v0     = %.2e [km/s]" % (v0/1e5))
print(" B0     = %.2e [G]" % (B0))
print(" p0     = %.2e [dyn]" % (p0))
print(" g0     = %.2e [dyn]" % (g0))

print("Model characteristics:")
print(" v_A(z=0)   = %.2e [km/s]" % (v_A_disk / 1e5))
print(" c_T(z=0)   = %.2e [km/s]" % (c_Td / 1e5))
print(" v_A(z=L)   = %.2e [km/s]" % (v_A_L / 1e5))
print(" c_T(z=z_d) = %.2e [km/s]" % (c_Ta / 1e5))
print(" v_k        = %.2e [km/s]" % (vk0 / 1e5))
print(" Mach       = %.1f" % (Mach))
print(" t_stop     = %.3f [t0] = %.2e [yr] = %.2e [d]" % (t_stop, t_stop*t0/year, t_stop*t0/day))
print(" v_max      = %.2e [km/s]" % (v_max / 1e5))

# ----- основные переменные ---------------------------------------------------

# число основных переменных
var_number = 6 
# консервативные переменные на шаге t^n
u_n = np.zeros((var_number, Ntot))

# консервативные переменные на шаге t^(n+1/2) (промежуточном)
u_s = np.zeros((var_number, Ntot))

# консервативные переменные на шаге t^(n+1)
u_n1 = np.zeros((var_number, Ntot))

# примитивные переменные на шаге t^n
pv_n  = np.zeros((var_number, Ntot))  # [rho, v_z, v_phi, b_phi, p]

# потоки на шаге t^n
F_n  = np.zeros((var_number, Ntot))
# потоки на шаге t^(n+1/2)
F_s  = np.zeros((var_number, Ntot))
# источники на шаге t^n
S_n  = np.zeros((var_number, Ntot))
# источники на шаге t^(n+1/2)
S_s  = np.zeros((var_number, Ntot))

# список векторов примитивных переменных
# pv  = [rho_n, vphi_n, vz_n, B_n, p_n]
# вектор полной скорости на сетке
vv = np.zeros((N))
# массив узлов сетки
zs = np.linspace(a, b, N)


# ----- построение расчетной сетки --------------------------------------------
dz = (b-a)/ (N-1)
dt = c*dz/(v_max/v0)
print(" dz  = %.2e [H]" % (dz))
print(" dt0 = %.2e [t0] = %.2e [yr] = %.2e [d]" % (dt, dt*t0/year, dt*t0/day))

zs[0] = a
for i in range (1, N):
    zs[i] = zs[i-1] + dz
    
# ----- функции начальных и граничных условий ----------------------------------

# начальное распределение безразмерной температуры
def T_IC(z):
    if z < z_d:
        return 1.0
    else:
        return Ta / T0
    
# начальное распределение безразмерной плотности
def rho_IC(z):
    if z < z_d:
        return rho_disk(z)
    else:
        #return rho_surf#rho_atm(z)
        return rho_atm(z)

# начальное распределние безразмерной скорости v_z
def v_IC(z):
    return 0.0

vphi_s = rotation_flag*v_phi0(r_cm, 1.0 * H) / v0
# начальное распределение безразмерной скорости v_phi
def vphi_IC(z):
    if z < z_d:
        return rotation_flag*v_phi0(r_cm, z * H) / v0
    else:
        return 0.0
    
# начальное распределение безразмерной Bphi    
def B_IC(z):
    return 0.0

# начальное распределение безразмерного давления p
def p_IC(z):
    return rho_IC(z) * T_IC(z)

def va_IC(z):
    return B0/np.sqrt(4*np.pi*rho_atm(z))

def p2c(cv, pv):
    """
    Перевод вектора примитивных переменных в вектор консервативных переменных

    Parameters
    ----------
    cv : массив
        вектор консервативных переменных.
    pv : массив
        вектор примитивных переменных.

    Returns
    -------
    None.

    """    
    # pv = {rho, vphi, vz, Bphi, p,va}
    
    cv[0, :] = pv[0, :] # rho
    cv[1:, ] = pv[0, :] * pv[1, :] # rho*v_phi
    cv[2, :] = pv[0, :] * pv[2, :] # rho*v_z
    cv[3, :] = pv[3, :] # Bphi
    cv[4, :] = pv[4, :] # p
    cv[5, :] = pv[5, :]
def c2p(cv, pv):
    """
     Перевод вектора консервативных переменных в вектор примитивных переменных

     Parameters
     
     ----------
     cv : массив
         вектор консервативных переменных.
     pv : массив
         вектор примитивных переменных.

     Returns
     -------
     None.

     """    
    pv[0, :] = cv[0, :] # rho
    pv[1, :] = cv[1, :] / cv[0, :] # v_phi = [rho*v_phi]/rho
    pv[2, :] = cv[2, :] / cv[0, :]  # v_z= [rho*v_z] / rho
    pv[3, :] = cv[3, :] # Bphi
    pv[4, :] = cv[4, :] # p
    pv[5, :] = cv[5, :]

# ----- установка начальных условий -------------------------------------------
def SetIC():
    # global t, u_n, pv
    global t, u_n, pv_n
    
    t = 0.0
    for i in range(Ngs, Ntot-Ngs): # без учета фиктивных ячеек: i=[Ngs, Ntot-Ngs) = [1, Ntot-2]
        # учитываем, что в массиве узлов zs нет координат фиктивных узлов    
        pv_n[0][i] = rho_IC(zs[i-Ngs])  # rho_n
        pv_n[1][i] = vphi_IC(zs[i-Ngs]) # vphi_n
        pv_n[2][i] = v_IC(zs[i-Ngs])    # vz_n
        pv_n[3][i] = B_IC(zs[i-Ngs])    # B_n
        pv_n[4][i] = p_IC(zs[i-Ngs])    # p_n
        pv_n[5][i] = va_IC(zs[i-Ngs])

    # перевод заданных начальных примитивных переменных в начальные консервативные
    p2c(u_n, pv_n)

# ----- функции потоков -------------------------------------------------------

# поток массы в каждой ячейке сетки
def F0(u):
    # = rho*v_z
    return u[2, :]

# поток импульса rho*v_phi в каждой ячейке сетки
def F1(u):
    # = rho*v_z*v_ph -  (2/beta)*Bz*Bphi
    return u[1, :] * u[2, :] / u[0, :] - 2 * u[3, :] * Bz / beta

# поток импульса rho*v_z в каждой ячейке сетки
def F2(u):
    # = rho*v_z^2 + p + B_phi^2/beta
    return (u[2, :])**2 / u[0, :] + u[4, :] + (u[3, :])**2/beta

# поток Bphi в каждой ячейке сетки
def F3(u):
    # = v_z*B_phi - v_phi*Bz
    return u[2, :] * u[3, :] / u[0, :] - u[1, :] * Bz / u[0, :]

# поток давления в каждой ячейке сетки
def F4(u):
    # = (rho*v_z) * p / rho
    return u[2, :] * u[4, :] / u[0, :]
#альвеновкая скорость
def F5(u):
    # =B0/(4*pi*rho)
    return B0/np.sqrt(4*np.pi*u[0, :])

# вектор-функция источников (правые части уравнений)
def Source(u, i, var_n):
    if var_n == 0:
        return 0.0
    elif var_n == 1:
        return 0.0
    elif var_n == 2:
        # безразмерная сила тяжести
        if (i == Ntot-Ngs-1): # этой точки нет в массиве zs
            return -g_z(r_cm, (zs[i-1] + dz)*H, Mstar)/g0 * u[0][i]
        else:
            return -g_z(r_cm, zs[i]*H, Mstar)/g0 * u[0][i]
        
    elif var_n == 3:
        return 0.0
    elif var_n == 4:
        return 0.0
    else:
        None
        
# установка граничных условий
def SetBC():
    global t, u_n1

    # левая граница
    rho_L  = 1.0 # 
    vphi_L = rotation_flag*vk0 / v0 #
    vz_L   = 0.0 #
    B_L    = 0.0 #
    p_L    = 1.0 # 
    va_L = 0.0
    
    # это значения на левой границе (i = Ngs = 1)
    u_n1[0][Ngs] = rho_L
    u_n1[1][Ngs] = rho_L * vphi_L
    u_n1[2][Ngs] = rho_L * vz_L
    u_n1[3][Ngs] = B_L
    u_n1[4][Ngs] = p_L
    u_n1[5][Ngs] = va_L
  

    # условия свободного втекания, посредством фиктивных ячеек
    u_n[:, Ntot-1] = u_n[:, Ntot-Ngs-2]    

# вычисление шага по времени
def UpdateTimeStep():
    global dt, v_max, v, v_n
    # флаг указывает, успешно ли выполнен расчет шага
    success = True
    # сообщение об успешности (неуспешности) расчета
    message = "Time step is successfully determined"

    for i in range(Ngs, Ntot-Ngs): # цикл без учета фиктивных ячеек
        # квадрат безразмерной альв. скорости
        vAvA = (2.0/beta) * (Bz**2 + pv_n[3, i]**2) / pv_n[0, i]
        
        # полная скорость: скорость газа в обоиз направлениях (v_z и v_phi) плюс магнитозвуковая скорость
        vv[i-Ngs] = abs(pv_n[2, i]) + abs(pv_n[1, i]) + np.sqrt(cTcT + vAvA)
        
    v_max = max(vv)
    i_max = np.argmax(vv)
    dt = c*dz/v_max

    # скорость больше 20000 км/с
    if ((v_max*v0/1e5) > 20000):
        success = False
        message = ("Velocity became unphysically large: v_max = %.3e [v0] = %.3e [km/s]\n   at z = %.2f [H], where rho = %.3e [rho0], B_z = %.3f [B0], B_phi = %.3e [B0]" % (v_max/1e5, v_max*v0/1e5, zs[i_max], pv_n[0, i_max], Bz, pv_n[3, i_max]))
    
    return [success, message]


# шаг по времени
def Step():
    global dt, u_n, u_s, u_n1, dz
    
    # источники на этапе предиктора
    S_n[2, Ngs:Ntot-Ngs-2:] = -g_z(r_cm, zs[Ngs:Ntot-Ngs-2:]*H, Mstar)/g0 * u_n[0, Ngs:Ntot-Ngs-2:]
    S_n[2, Ntot-Ngs-1] = -g_z(r_cm, (zs[Ntot-Ngs-2] + dz)*H, Mstar)/g0 * u_n[0, Ntot-Ngs-1]

    # потоки на этапе предиктора
    F_n[0, :] = F0(u_n)
    F_n[1, :] = F1(u_n)
    F_n[2, :] = F2(u_n)
    F_n[3, :] = F3(u_n)
    F_n[4, :] = F4(u_n)
    F_n[5, :] = F5(u_n)

    # Метод Л-В, этап предиктора
    for i in range(Ngs, Ntot-Ngs):
        u_s[:, i] =  0.5 * (u_n[:, i+1] + u_n[:, i]) - 0.5*(dt/dz)*(F_n[:, i+1] - F_n[:, i]) + S_n[:, i]*dt
            
    # источники на этапе корректора
    S_s[2, Ngs+1:Ntot-Ngs-2:] = -g_z(r_cm, zs[Ngs+1:Ntot-Ngs-2:]*H, Mstar)/g0 * u_s[0, Ngs+1:Ntot-Ngs-2:]
    S_s[2, Ntot-Ngs-1] = -g_z(r_cm, (zs[Ntot-Ngs-2] + dz)*H, Mstar)/g0 * u_s[0, Ntot-Ngs-1]

    # потоки на этапе корректора
    F_s[0, :] = F0(u_s)
    F_s[1, :] = F1(u_s)
    F_s[2, :] = F2(u_s)
    F_s[3, :] = F3(u_s)
    F_s[4, :] = F4(u_s)
    F_s[5, :] = F5(u_s)
    
    # Метод Л-В, этап корректора
    for i in range (Ngs + 1, Ntot-Ngs):
        u_n1[:, i] = u_n[:, i] - (dt/dz) * (F_s[:, i] - F_s[:, i-1]) + 0.5 * dt * (S_s[:, i] + S_n[:, i]) + eps * dt * (u_n[:, i+1] - 2*u_n[:, i] + u_n[:, i-1])
            
    # построить на основе решения вектор примитивных переменных
    c2p(u_n1, pv_n)

# обновление НУ
def UpdateIC():
    global t, u_n, u_n1

    # копируем значения внутри сетки, без учета фиктивных ячеек
    u_n[:, Ngs:Ntot-Ngs:] = u_n1[:, Ngs:Ntot-Ngs:]

def SaveData(n, t_n):
    data = []
    data.append(zs)
    
    if n == 0:
        c2p(u_n, pv_n)
        for var_n in range(var_number):
            data.append(pv_n[var_n, Ngs:Ntot-Ngs:])
    else:
        c2p(u_n1, pv_n)
        for var_n in range(var_number):
            data.append(pv_n[var_n, Ngs:Ntot-Ngs:])

    # Correctly create fmt with 6 specifiers
    fmt = ('%.3e',) * (var_number + 1)  

    np.savetxt("./out/data" + str(n) + ".dat", np.array(data).transpose(), fmt=fmt, header=("%.2e" % t_n), comments="")


print("Press Enter to start a simulation")
input()

n = 0 # номер шага по времени
SetIC()
# Сохранение НУ
SaveData(n, t)

n = n + 1
start = timer()
while t <= t_stop:
    [contin, message] = UpdateTimeStep()  
    # выход из цикла и сохранение результатов, если что-то пошло не так (TO DO)
    if(contin == False):
        print("Exiting because", message)
        SaveData(n, t)
        break

    SetBC()
    
    Step()
    # вывод n, t, dt на экран и сохранение результатов каждые ... шагов
    if ((n % 100) == 0):
        print(  "step No ", n)
        print(  " t     = %.2e [t0] = %.1e [s]" % (t, t*t0))
        print(  " dt    = %.2e [t0] = %.1e [s]" % (dt, dt*t0))
        print(  " v_max = %.2e [km/s]" % (v_max*v0/1e5))
        SaveData(n, t)
    UpdateIC()  
    t += dt
    n += 1

end = timer()
# время расчета, сек
run_time = end - start
print(u'Run time = %.3e [s]' % (run_time))
print("Total number of steps = ", n)
SaveData(n, t)

# ------ сохранение параметров модели и расчета в файл ------------------------------
summary_file = open("./out6/model_summary.txt", "w")
summary_file.write("----- Scales -----\n")
summary_file.write("  r_0   = %.3e [au]\n" % (H/au))
summary_file.write("  t_0   = %.3e [yr] = %.3f [d]\n" % (t0/year, t0/day))
summary_file.write("  v_0   = %.3e [km/s]\n" % (v0/1e5))
summary_file.write("  rho_0 = %.3e [g/cm^3]\n" % (rho0))
summary_file.write("  p_0   = %.3e [dyn]\n" % (p0))
summary_file.write("  B_0   = %.3e [G]\n" % (B0))
summary_file.write("  g_0   = %.3e [dyn]\n" % (g0))

summary_file.write("\n----- Star and disk characteristics -----\n")
summary_file.write("  M_star   = %.2e [M_sun]\n" % (Mstar))
summary_file.write("  rho(z=0) = %.2e [g/cm^3]\n" % (rho0))
summary_file.write("  T(z=0)   = %.0f [K]\n" % (T0))
summary_file.write("  H        = %.2e [au]\n" % (H / au))
summary_file.write("  T_atm    = %.0f [K]\n" % (Ta))
summary_file.write("  n_ISM    = %.2e [cm^(-3)]\n" % (n_ISM))

summary_file.write("\n----- Model parameters -----\n")
summary_file.write("  r    = %.2f [au]\n" % (r_cm / au))
summary_file.write("  beta = %.2e\n" % (beta))
summary_file.write("  Mach = %.1f\n" % (Mach))

summary_file.write("\n----- Run parameters -----\n")
summary_file.write("  t_stop   = %.2f [t0] = %.2e [yr] \n" % (t_stop, t_stop*t0/year))
summary_file.write("  Courant  = %.2f\n" % (c))
summary_file.write("  N        = %.0d\n" % (N))
summary_file.write("  eps_visc = %.0f\n" % (eps))
summary_file.write("  run time = %.2f [sec] = %.2f [min]\n" % (run_time, run_time/60))

summary_file.close()