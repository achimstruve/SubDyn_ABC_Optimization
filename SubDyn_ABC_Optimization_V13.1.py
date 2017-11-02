from __future__ import division
import numpy as np
import time
from random import *
import math
import sys
import shutil
import os
from Get_Coordinates import get_joints_and_members, get_brace_coords
from Tower_Top_Masses import Masses_and_Inertias
from Python_FunctionsV1 import skew_matrix, mag, totuple, fs

start_dt = time.strftime("%Y.%m.%d") + "_" + time.strftime("%H_%M_%S")

# Turbine Specifications
root_main = "D:\Struve\Promotion\Calculation\SubDyn_Automated\SubDyn_ABC_Optimization\\"
current_folder = "\\SubDyn_Inputs_" + start_dt + "\\"
root = root_main+current_folder
root_out = root + "\\Lattice_Rotatable_output\\"
if not os.path.exists(root):
    os.makedirs(root)
if not os.path.exists(root_out):
    os.makedirs(root_out)
if not os.path.exists(root + "ANSYS_Calc\\"):
    os.makedirs(root + "ANSYS_Calc\\")


# optimization parameters
n_employed_bees = 48 # minimum two in order to chose another bees parameters
n_onlooker_bees = 48 # minimum two in order to chose another bees parameters
n_scout_bees = 1

initial_conditions = False
initial_cycle = 0

if not initial_conditions:
    initial_cycle = 0

 # DER ABMINDERUNGSFAKTOR rho bezieht sich nur auf den angeschlossenen Winkel! Dies ist noch nicht implementiert!
# Die Schlankheit von Eckstielen sollte den Wert lambda=120 nicht ueberschreiten DIN EN 1993-3-1_2010 H.2
# Die Schlankheit von primaeren Fuellstaeben sollte den Wert lambda=180 und von sekundaeren Staben von lambda = 250 nicht ueberschreiten

# Tower Parameters
Tower_Height = 87.6
"""
a_Top_max = 8
a_Top_min = 3.5
b_Top_max = 8
b_Top_min = 3.5
a_Bot_max = 30
a_Bot_min = 8
b_Bot_max = 30
b_Bot_min = 8
"""
a_Top = 10
b_Top = 10
a_Bot = 20
b_Bot = 20
#sF_min = 0.5
#sF_max = 1.2



Nseg_min = 7
Nseg_max = 18

alpha_y_min = -20
alpha_y_max = 0
alpha_y_min = alpha_y_min* np.pi / 180.0
alpha_y_max = alpha_y_max* np.pi / 180.0
alpha_x = 0
alpha_x = alpha_x * np.pi / 180.0

tolerance = .0001

# Define cross-sections with properties: PropSetID, YoungE, ShearG, MatDens, XsecA, XsecAx, XsecAy, XsecAxy, Xsecxs, Xsecys, XsecJxx, XsecJyy, XsecJxy, XsecJ0
# Define L-Profile Dimensions (length of vertical leg, length of horizontal leg, thickness of vertical leg, thickness of horizontal leg )
Llmax = 1.2
Llmin = 0.15
tlmax = 0.055
tlmin = 0.008
Lbmax = 0.8
Lbmin = 0.06
tbmax = 0.02
tbmin = 0.006

n_cycles_max = 20000
eff_tol = 1e-6 # tolerance for effective cross-section convergence

# Materail Parameters
g = 9.81 # gravitational acceleration / m/s^2
E = 0.210000E+12
G = 8.076923e+10
nu = E/(2*G) - 1
rho_steel = 7850 * 1.01
fyk = 355e6
gamma_m = 1.1
gamma_buckle = 1.1
beta_buckling = 0.9 # crictical member buckling length can be reduced by 0.9 due to EC 1993-1-1_2010, p.91
alpha_b = 0.34 # imperfection factor
alpha_LT = 0.76 # imperfection for torsional buckling according to EC 1993-1-1_2010 page 68 Table 6.3 and 6.4
beta_v = 1   # fork bearing on both sides
beta_theta= 1 # fork bearing on both sides
buckle_min = gamma_m

# Turbine Parameters
mass_Nacelle = 240000
mass_Rot = 110000
mass_blade = 17740
mass_hub = 56780
mass_NREL_tower = 347460

inertia_blade_root = 11776047
inertia_hub_CM = 115926  # because the hub is assumed to be a thin spherical shell, the hub inertia is the same for each axis (see the NREL 5MW definition document)
inertia_nacelle_yaw = 2607890
inertia_generator_hss = 534116

D_tower_top = 3.87

Overhang = 5
Twr2Shft = 1.96256
NacCMxn = 1.9
NacCMzn = 1.75
CM_blade_root = 20.475
precone = 2.5 # deg
CM_blade_downwind = Overhang + np.sin(precone * np.pi / 180) * CM_blade_root

# frequency constraints due to operation
f1in = 6.9 / 60 * 0.95
f1out = 12.1 / 60 * 1.05
f3in = 6.9 / 60 * 0.95 * 3
f3out = 12.1 / 60 * 1.05 * 3
f_min = f3out * 1.05

### Set SubDyn parameters
NDiv = 1
JDampings = 50
JDampings_static = 50
NReact = 4
NInterf = 7
NCmass = 2
NPropSets = 0
SSoutList_Lines = 4
L_Ap = 1

NSteps1 = 150 # set the amount of time steps for the first analysis (natural frequency, mass, and equivalent stiffness matrix)
TimeInterval1 = 0.01 # set the time step width for the first analysis (natural frequency, mass, and equivalent stiffness matrix)
NSteps2 = 150 # set the amount of time steps for the second analysis (load case)
TimeInterval2 = 0.01 # set the time step width for the second analysis (load case)

# Define Member Outputs
#MOutputs_a = np.array([['   1        2       1, 2'],['   2        2       1, 2'],['   3        2       1, 2'],['   4        2       1, 2']])
MOutputs_a = np.array([])
NMOutputs = len(MOutputs_a)
Out_Param_a = np.array([['"ReactFXss, ReactFYss, ReactFZss, ReactMXss, ReactMYss, ReactMZss"    -Base reactions (forces onto SS structure)'], \
                        ['"IntfFXss,  IntfFYss,  IntfFZss,  IntfMXss, IntfMYss, IntfMZss"       -Interface reactions (forces onto SS structure)']])
"""
Out_Param_a = np.array([['"ReactFXss, ReactFYss, ReactFZss, ReactMXss, ReactMYss, ReactMZss"    -Base reactions (forces onto SS structure)'], \
                        ['"IntfFXss,  IntfFYss,  IntfFZss,  IntfMXss, IntfMYss, IntfMZss"       -Interface reactions (forces onto SS structure)'], \
                        ['"M1N1FKxe, M1N1FKye, M1N1FKze, M1N1MKxe, M1N1MKye, M1N1MKze"'], \
                        ['"M1N2FKxe, M1N2FKye, M1N2FKze, M1N2MKxe, M1N2MKye, M1N2MKze"'], \
                        ['"M2N1FKxe, M2N1FKye, M2N1FKze, M2N1MKxe, M2N1MKye, M2N1MKze"'], \
                        ['"M2N2FKxe, M2N2FKye, M2N2FKze, M2N2MKxe, M2N2MKye, M2N2MKze"'], \
                        ['"M3N1FKxe, M3N1FKye, M3N1FKze, M3N1MKxe, M3N1MKye, M3N1MKze"'], \
                        ['"M3N2FKxe, M3N2FKye, M3N2FKze, M3N2MKxe, M3N2MKye, M3N2MKze"'], \
                        ['"M4N1FKxe, M4N1FKye, M4N1FKze, M4N1MKxe, M4N1MKye, M4N1MKze"'], \
                        ['"M4N2FKxe, M4N2FKye, M4N2FKze, M4N2MKxe, M4N2MKye, M4N2MKze"']])

"M1N1FKxe, M1N1FKye, M1N1FKze, M1N1MKxe, M1N1MKye, M1N1MKze"
"M1N2FKxe, M1N2FKye, M1N2FKze, M1N2MKxe, M1N2MKye, M1N2MKze"
"M2N1FKxe, M2N1FKye, M2N1FKze, M2N1MKxe, M2N1MKye, M2N1MKze"
"M2N2FKxe, M2N2FKye, M2N2FKze, M2N2MKxe, M2N2MKye, M2N2MKze"
"M3N1FKxe, M3N1FKye, M3N1FKze, M3N1MKxe, M3N1MKye, M3N1MKze"
"M3N2FKxe, M3N2FKye, M3N2FKze, M3N2MKxe, M3N2MKye, M3N2MKze"
"M4N1FKxe, M4N1FKye, M4N1FKze, M4N1MKxe, M4N1MKye, M4N1MKze"
"M4N2FKxe, M4N2FKye, M4N2FKze, M4N2MKxe, M4N2MKye, M4N2MKze"
"""
SSoutList_Lines = 1 + len(Out_Param_a)

# Define DLCs with each row for each DLC (loads in N and Nm)
TP_Load_Case1 = np.array([1.64e3, 3.73, -4.76e3, 6.17e3, 3.41e3, 2.21e3]) * 1e3 # maximum Fx
TP_Load_Case2 = np.array([-3.01e2, 5.76e1, -4.65e3, -1.94e2, -1.33e3, -9.54e2]) * 1e3 # minimum Fx
TP_Load_Case3 = np.array([5.58e2, -3.69e2, -4.70e3, 4.42e3, -2.81e3, -6.35e2]) * 1e3 # maximum abs(Fy)
TP_Load_Case4 = np.array([4.76e2, 3.57e1, -4.56e3, 4.41e3, -8.64e2, -1.23e4]) * 1e3 # maximum abs(Mz)
TP_Load_Case_max = np.array([1.64e3, -3.69e2, -5.41e3, -8.42e3, 1.51e4, -1.23e4]) * 1e3 # each absolute maximum value from each load component

n_DLC = 4
TP_Load_Cases = np.zeros([n_DLC, 6])
TP_Load_Cases = np.array([TP_Load_Case1, TP_Load_Case2, TP_Load_Case3, TP_Load_Case4])

### Functions ###
### Calculate bending normal stresses
def normal_stress(Fz,Mx,My,A,Jxx,Jyy,Jxy,x,y):
    sigma_a = Fz / A
    sigma_b = ((-Mx*Jxy - My*Jxx) * x + (Mx*Jyy + My*Jxy) * y) / (Jxx*Jyy - Jxy**2)
    sigma = sigma_a + sigma_b
    return sigma

def effective_cross_section(psiv1, psih1, rhov1_eff, rhoh1_eff, lv_m, lh_m, tv_m, th_m, sigma_vt1, sigma_vbhl1, sigma_hr1):
    #print "psiv: ", round(psiv1,3), " psih: ", round(psih1,3), " rhov_eff: ", round(rhov1_eff,3), " rhoh_eff: ", round(rhoh1_eff,3), " sigma_vt: ",fs(sigma_vt1,4), " sigma_vbhl: ", fs(sigma_vbhl1,4), " sigma_hr: ", fs(sigma_hr1,4)
    if psiv1 <= 1 and psiv1 >= 0:
        #print "c1 ",
        b_effv1 = rhov1_eff * (lv_m-th_m)
    if psiv1 < 0:
        #print "c2 ",
        if sigma_vt1 > sigma_vbhl1:
            #print "c2.1",
            b_cv1 = (lv_m-th_m) / (1-psiv1)
            b_ev1 = rhov1_eff * b_cv1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
            b_effv1 = (lv_m-th_m) - b_cv1 + b_ev1
    if (psih1 <= 1 and psih1 >= 0) and not (sigma_vt1 <= sigma_vbhl1 and psiv1 < 0):
        #print "c3 ",
        b_effh1 = rhoh1_eff * (lh_m-tv_m)                           
        A_eff1 = b_effv1 * tv_m + b_effh1 * th_m + tv_m * th_m
        cx_eff1 = b_effh1 * th_m * (tv_m/2 + b_effh1/2) / A_eff1
        cy_eff1 = b_effv1 * tv_m * (th_m/2 + b_effv1/2) / A_eff1
        Jxx_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_effv1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + (cy_eff1-(th_m/2 + b_effv1/2))**2 * b_effv1 * tv_m
        Jyy_eff1 = b_effh1**3 * th_m / 12 + tv_m**3 * th_m / 12 + b_effv1 * tv_m**3 / 12 + cx_eff1**2 * (b_effv1 * tv_m + tv_m * th_m) + (cx_eff1-(tv_m/2 + b_effh1/2))**2 * b_effh1 * th_m
        Jxy_eff1 = -cx_eff1 * (th_m/2 + b_effv1/2 - cy_eff1) * b_effv1 * tv_m - cx_eff1 * (- cy_eff1) * tv_m * th_m - cy_eff1 * (tv_m/2 + b_effh1/2 - cx_eff1) * b_effh1 * th_m
    if psih1 < 0 and not (sigma_vt1 <= sigma_vbhl1 and psiv1 < 0):
        #print "c4 ",
        if sigma_vbhl1 <= sigma_hr1:
            #print "c4.1 ",
            b_ch1 = (lh_m-tv_m) / (1-psih1)
            b_eh1 = rhoh1_eff * b_ch1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
            b_effh1 = (lh_m-tv_m) - b_ch1 + b_eh1
            A_eff1 = b_effv1 * tv_m + b_effh1 * th_m + tv_m * th_m
            cx_eff1 = b_effh1 * th_m * (tv_m/2 + b_effh1/2) / A_eff1
            cy_eff1 = b_effv1 * tv_m * (th_m/2 + b_effv1/2) / A_eff1
            Jxx_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_effv1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + (cy_eff1-(th_m/2 + b_effv1/2))**2 * b_effv1 * tv_m
            Jyy_eff1 = b_effh1**3 * th_m / 12 + tv_m**3 * th_m / 12 + b_effv1 * tv_m**3 / 12 + cx_eff1**2 * (b_effv1 * tv_m + tv_m * th_m) + (cx_eff1-(tv_m/2 + b_effh1/2))**2 * b_effh1 * th_m
            Jxy_eff1 = -cx_eff1 * (th_m/2 + b_effv1/2 - cy_eff1) * b_effv1 * tv_m - cx_eff1 * (- cy_eff1) * tv_m * th_m - cy_eff1 * (tv_m/2 + b_effh1/2 - cx_eff1) * b_effh1 * th_m
        if sigma_vbhl1 > sigma_hr1:
            #print "c4.2 ",
            b_ch1 = (lh_m-tv_m) / (1-psih1)
            b_eh1 = rhoh1_eff * b_ch1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
            b_th1 = (lh_m-tv_m) - b_ch1                              
            b_effh1 = b_th1 + b_eh1
            A_eff1 = b_effv1 * tv_m + b_effh1 * th_m + tv_m * th_m
            cx_eff1 = (b_eh1 * th_m * (tv_m/2 + b_eh1/2) + b_th1 * th_m * (tv_m/2 + b_ch1 + b_th1/2)) / A_eff1
            cy_eff1 = b_effv1 * tv_m * (th_m/2 + b_effv1/2) / A_eff1
            Jxx_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_effv1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + (cy_eff1-(th_m/2 + b_effv1/2))**2 * b_effv1 * tv_m
            Jyy_eff1 = b_eh1**3 * th_m / 12 + b_th1**3 * th_m / 12 + tv_m**3 * th_m / 12 + b_effv1 * tv_m**3 / 12 + cx_eff1**2 * (b_effv1 * tv_m + tv_m * th_m) + ((tv_m/2 + b_ch1 + b_th1/2) - cx_eff1)**2 * b_th1 * th_m + ((tv_m/2 + b_eh1/2) - cx_eff1)**2 * b_eh1 * th_m
            Jxy_eff1 = -cx_eff1 * (th_m/2 + b_effv1/2 - cy_eff1) * b_effv1 * tv_m - cx_eff1 * (- cy_eff1) * tv_m * th_m - cy_eff1 * (tv_m/2 + b_eh1/2 - cx_eff1) * b_eh1 * th_m - cy_eff1 * (tv_m/2 + b_ch1 + b_th1/2 - cx_eff1) * b_th1 * th_m   
    if sigma_vt1 <= sigma_vbhl1 and psiv1 < 0:
        #print "c5 ",
        if psih1 <= 1 and psih1 >= 0:
            #print "c5.1 ",
            b_effh1 = rhoh1_eff * (lh_m-tv_m)
        if psih1 < 0 and sigma_vbhl1 <= sigma_hr1:
            #print "c5.2 ",
            b_ch1 = (lh_m-tv_m) / (1-psih1)
            b_eh1 = rhoh1_eff * b_ch1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
            b_effh1 = (lh_m-tv_m) - b_ch1 + b_eh1
        if (psih1 < 0 and sigma_vbhl1 <= sigma_hr1) or (psih1 <= 1 and psih1 >= 0):
            #print "c5.3 ",
            b_cv1 = (lv_m-th_m) / (1-psiv1)
            b_ev1 = rhov1_eff * b_cv1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
            b_tv1 = (lh_m-tv_m) - b_cv1
            b_effv1 = b_tv1 + b_ev1
            A_eff1 = b_effv1 * tv_m + b_effh1 * th_m + tv_m * th_m
            cx_eff1 = b_effh1 * th_m * (tv_m/2 + b_effh1/2) / A_eff1
            cy_eff1 = (b_ev1 * tv_m * (th_m/2 + b_ev1/2) + b_tv1 * tv_m * (th_m/2 + b_cv1 + b_tv1/2)) / A_eff1
            Jxx_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_tv1**3 * tv_m / 12 + b_ev1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + ((th_m/2 + b_cv1 + b_tv1/2) - cy_eff1)**2 * b_tv1 * tv_m + ((th_m/2 + b_ev1/2) - cy_eff1)**2 * b_ev1 * tv_m
            Jyy_eff1 = b_effh1**3 * th_m / 12 + tv_m**3 * th_m / 12 + b_effv1 * tv_m**3 / 12 + cx_eff1**2 * (b_effv1 * tv_m + tv_m * th_m) + (cx_eff1-(tv_m/2 + b_effh1/2))**2 * b_effh1 * th_m
            Jxy_eff1 = -cx_eff1 * (th_m/2 + b_ev1/2 - cy_eff1) * b_ev1 * tv_m -cx_eff1 * (th_m/2 + b_cv1 + b_tv1/2 - cy_eff1) * b_tv1 * tv_m - cx_eff1 * (- cy_eff1) * tv_m * th_m - cy_eff1 * (tv_m/2 + b_effh1/2 - cx_eff1) * b_effh1 * th_m
        if psih1 < 0 and sigma_vbhl1 > sigma_hr1:
            #print "c5.4 "
            b_ch1 = (lh_m-tv_m) / (1-psih1)
            b_eh1 = rhoh1_eff * b_ch1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
            b_th1 = (lh_m-tv_m) - b_ch1                              
            b_effh1 = b_th1 + b_eh1
            b_cv1 = (lv_m-th_m) / (1-psiv1)
            b_ev1 = rhov1_eff * b_cv1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
            b_tv1 = (lh_m-tv_m) - b_cv1
            b_effv1 = b_tv1 + b_ev1
            A_eff1 = b_effv1 * tv_m + b_effh1 * th_m + tv_m * th_m
            cx_eff1 = (b_eh1 * th_m * (tv_m/2 + b_eh1/2) + b_th1 * th_m * (tv_m/2 + b_ch1 + b_th1/2)) / A_eff1
            cy_eff1 = (b_ev1 * tv_m * (th_m/2 + b_ev1/2) + b_tv1 * tv_m * (th_m/2 + b_cv1 + b_tv1/2)) / A_eff1
            Jxx_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_tv1**3 * tv_m / 12 + b_ev1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + ((th_m/2 + b_cv1 + b_tv1/2) - cy_eff1)**2 * b_tv1 * tv_m + ((th_m/2 + b_ev1/2) - cy_eff1)**2 * b_ev1 * tv_m
            Jyy_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_tv1**3 * tv_m / 12 + b_ev1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + ((th_m/2 + b_cv1 + b_tv1/2) - cy_eff1)**2 * b_tv1 * tv_m + ((th_m/2 + b_ev1/2) - cy_eff1)**2 * b_ev1 * tv_m
            Jxy_eff1 = -cx_eff1 * (th_m/2 + b_ev1/2 - cy_eff1) * b_ev1 * tv_m -cx_eff1 * (th_m/2 + b_cv1 + b_tv1/2 - cy_eff1) * b_tv1 * tv_m - cx_eff1 * (- cy_eff1) * tv_m * th_m - cy_eff1 * (tv_m/2 + b_eh1/2 - cx_eff1) * b_eh1 * th_m - cy_eff1 * (tv_m/2 + b_ch1 + b_th1/2 - cx_eff1) * b_th1 * th_m 
    
    if sigma_vt1 <= 0 and sigma_vbhl1 <= 0:
        if sigma_hr1 <= 0:
            b_effh1 = (lh_m-tv_m)
            b_effv1 = (lv_m-th_m)                         
            A_eff1 = b_effv1 * tv_m + b_effh1 * th_m + tv_m * th_m
            cx_eff1 = b_effh1 * th_m * (tv_m/2 + b_effh1/2) / A_eff1
            cy_eff1 = b_effv1 * tv_m * (th_m/2 + b_effv1/2) / A_eff1
            Jxx_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_effv1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + (cy_eff1-(th_m/2 + b_effv1/2))**2 * b_effv1 * tv_m
            Jyy_eff1 = b_effh1**3 * th_m / 12 + tv_m**3 * th_m / 12 + b_effv1 * tv_m**3 / 12 + cx_eff1**2 * (b_effv1 * tv_m + tv_m * th_m) + (cx_eff1-(tv_m/2 + b_effh1/2))**2 * b_effh1 * th_m
            Jxy_eff1 = -cx_eff1 * (th_m/2 + b_effv1/2 - cy_eff1) * b_effv1 * tv_m - cx_eff1 * (- cy_eff1) * tv_m * th_m - cy_eff1 * (tv_m/2 + b_effh1/2 - cx_eff1) * b_effh1 * th_m

        if sigma_hr1 > 0:
            b_ch1 = (lh_m-tv_m) / (1-psih1)
            b_eh1 = rhoh1_eff * b_ch1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
            b_effh1 = (lh_m-tv_m) - b_ch1 + b_eh1    
            b_effv1 = (lv_m-th_m)                         
            A_eff1 = b_effv1 * tv_m + b_effh1 * th_m + tv_m * th_m
            cx_eff1 = b_effh1 * th_m * (tv_m/2 + b_effh1/2) / A_eff1
            cy_eff1 = b_effv1 * tv_m * (th_m/2 + b_effv1/2) / A_eff1
            Jxx_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_effv1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + (cy_eff1-(th_m/2 + b_effv1/2))**2 * b_effv1 * tv_m
            Jyy_eff1 = b_effh1**3 * th_m / 12 + tv_m**3 * th_m / 12 + b_effv1 * tv_m**3 / 12 + cx_eff1**2 * (b_effv1 * tv_m + tv_m * th_m) + (cx_eff1-(tv_m/2 + b_effh1/2))**2 * b_effh1 * th_m
            Jxy_eff1 = -cx_eff1 * (th_m/2 + b_effv1/2 - cy_eff1) * b_effv1 * tv_m - cx_eff1 * (- cy_eff1) * tv_m * th_m - cy_eff1 * (tv_m/2 + b_effh1/2 - cx_eff1) * b_effh1 * th_m

    if sigma_vt1 >= 0 and sigma_vbhl1 <= 0 and sigma_hr1 <= 0:
        b_effh1 = (lh_m-tv_m)
        b_cv1 = (lv_m-th_m) / (1-psiv1)
        b_ev1 = rhov1_eff * b_cv1   # this variable is equal to b_eff in EC 1993-1-5, but I use b_eff as the width inclusive the tension region
        b_effv1 = (lv_m-th_m) - b_cv1 + b_ev1            
        A_eff1 = b_effv1 * tv_m + b_effh1 * th_m + tv_m * th_m
        cx_eff1 = b_effh1 * th_m * (tv_m/2 + b_effh1/2) / A_eff1
        cy_eff1 = b_effv1 * tv_m * (th_m/2 + b_effv1/2) / A_eff1
        Jxx_eff1 = b_effh1 * th_m**3 / 12 + tv_m * th_m**3 / 12 + b_effv1**3 * tv_m / 12 + cy_eff1**2 * (b_effh1 * th_m + tv_m * th_m) + (cy_eff1-(th_m/2 + b_effv1/2))**2 * b_effv1 * tv_m
        Jyy_eff1 = b_effh1**3 * th_m / 12 + tv_m**3 * th_m / 12 + b_effv1 * tv_m**3 / 12 + cx_eff1**2 * (b_effv1 * tv_m + tv_m * th_m) + (cx_eff1-(tv_m/2 + b_effh1/2))**2 * b_effh1 * th_m
        Jxy_eff1 = -cx_eff1 * (th_m/2 + b_effv1/2 - cy_eff1) * b_effv1 * tv_m - cx_eff1 * (- cy_eff1) * tv_m * th_m - cy_eff1 * (tv_m/2 + b_effh1/2 - cx_eff1) * b_effh1 * th_m

    return A_eff1, cx_eff1, cy_eff1, Jxx_eff1, Jyy_eff1, Jxy_eff1, b_effv1, b_effh1

def cross_sectional_yield_strength(lv_m, lh_m, tv_m, th_m, cx_m, cy_m, epsilon, Mx, My):
    
    lambda_bar_pvA = (lv_m - th_m) / (tv_m * 28.4 * epsilon * np.sqrt(0.43))
    lambda_bar_phA = (lh_m - tv_m) / (th_m * 28.4 * epsilon * np.sqrt(0.43))
    if lambda_bar_pvA <= 0.748:
        rho_vA = 1
    if lambda_bar_vA > 0.748:
        rho_vA = (lambda_bar_vA - 0.188) / lambda_bar_vA**2
        if rho_vA > 1:
            rho_vA = 1
    if lambda_bar_phA <= 0.748:
        rho_hA = 1
    if lambda_bar_hA > 0.748:
        rho_hA = (lambda_bar_hA - 0.188) / lambda_bar_hA**2
        if rho_hA > 1:
            rho_hA = 1
    
    Aeff = rho_vA * lv_m * tv_m + tv_m * th_m + rho_hA * lh_m * th_m

    cx_eff = rho_hA * lh_m * th_m * (tv_m/2 + rho_hA * lh_m/2) / A_eff
    cy_eff = rho_vA * lv_m * tv_m * (th_m/2 + rho_vA * lv_m/2) / A_eff
    
    return Aeff, cx_eff, cy_eff

### Bee functions and Classes
def weighted_choice(choices):
   total = 0
   for i in range(len(choices[:,1])):
       total += choices[i,1]
   r = uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"

def calc_standard_deviation(bee_list, bee_type):
    data_row = []
    for i in range(len(bee_list[:,1])):
        if bee_list[i,1].bee_type == "employed bee":
            data_row.append(bee_list[i,1].fitness)
    X_line = np.mean(data_row)
    n = len(data_row)
    summe = 0
    for i in range(len(data_row)):
        X_i = data_row[i]
        summe = summe + (X_i - X_line)**2
    if (n - 1) <> 0:
        S = np.sqrt(1 / (n - 1) * summe)
    if (n - 1) == 0:
        S = 0
    return S, len(data_row)

def prep_obj_func0(parameter_set_inst, bee_ID, cycle, bee):
    # Create the model of this bee in SubDyn

    Tower_Parameters = parameter_set_inst.parameters
    bee.Nseg = int(Tower_Parameters[len(Tower_Parameters)-3, 1])
    bee.alpha_y = Tower_Parameters[len(Tower_Parameters)-2, 1]
    bee.sF = Tower_Parameters[len(Tower_Parameters)-1, 1]
    bee.b_Top = bee.a_Top * bee.sF
    bee.b_Bot = bee.a_Bot * bee.sF

    # Get other SubDyn relevant parameters
    bee.NJoints = (bee.Nseg * 2 + 1) * 4 + 3 # + 3 because of the three joints for the tower top configuration for rotor, nacelle and tower top stiffening
    bee.NMembers = bee.Nseg * 20 + 4 + 5 # + 4 because of the tower top stiffening cross members and + 5 because of the attachment members for the rotor and the nacelle CMs
    bee.Nmodes = 20

    # Member Property Shedule
    Member_Props = np.zeros([bee.NMembers, 2])
    NXPropSet_a_IDs = np.arange(1,3 * bee.Nseg + 2)
    z = 0
    for i in range(len(Member_Props[:,0])):
        # leg 1
        if i < 1 * bee.Nseg:
            Member_Props[i,0] = NXPropSet_a_IDs[z]
            Member_Props[i,1] = NXPropSet_a_IDs[z]
            z += 3
        
        # leg 2
        if i >= 1 * bee.Nseg and i < 2 * bee.Nseg:
            if i == 1 * bee.Nseg:
                z = 0
            Member_Props[i,0] = NXPropSet_a_IDs[z]
            Member_Props[i,1] = NXPropSet_a_IDs[z]
            z += 3
        
        # leg 3
        if i >= 2 * bee.Nseg and i < 3 * bee.Nseg:
            if i == 2 * bee.Nseg:
                z = 0
            Member_Props[i,0] = NXPropSet_a_IDs[z]
            Member_Props[i,1] = NXPropSet_a_IDs[z]
            z += 3
        
        # leg 4
        if i >= 3 * bee.Nseg and i < 4 * bee.Nseg:
            if i == 3 * bee.Nseg:
                z = 0
            Member_Props[i,0] = NXPropSet_a_IDs[z]
            Member_Props[i,1] = NXPropSet_a_IDs[z]
            z += 3
        
        # Brace xp (b side)
        if i == 4 * bee.Nseg:
            z = 2 # index of corresponding property set in NXPropSet_a
            z2 = 0 # counter for each member in brace
            for j in range(bee.Nseg):        
                Member_Props[i + z2, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 1, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 1, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 2, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 2, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 3, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 3, 1] = NXPropSet_a_IDs[z]
                z += 3
                z2 += 4
        
        # Brace yp (a side)
        if i == 4 * bee.Nseg + 1 * (4 * bee.Nseg):
            z = 1 # index of corresponding property set in NXPropSet_a
            z2 = 0 # counter for each member in brace
            for j in range(bee.Nseg):        
                Member_Props[i + z2, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 1, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 1, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 2, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 2, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 3, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 3, 1] = NXPropSet_a_IDs[z]
                z += 3
                z2 += 4
    
        # Brace xm (b side)
        if i == 4 * bee.Nseg + 2 * (4 * bee.Nseg):
            z = 2 # index of corresponding property set in NXPropSet_a
            z2 = 0 # counter for each member in brace
            for j in range(bee.Nseg):        
                Member_Props[i + z2, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 1, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 1, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 2, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 2, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 3, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 3, 1] = NXPropSet_a_IDs[z]
                z += 3
                z2 += 4
    
        # Brace ym (a side)
        if i == 4 * bee.Nseg + 3 * (4 * bee.Nseg):
            z = 1 # index of corresponding property set in NXPropSet_a
            z2 = 0 # counter for each member in brace
            for j in range(bee.Nseg):        
                Member_Props[i + z2, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 1, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 1, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 2, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 2, 1] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 3, 0] = NXPropSet_a_IDs[z]
                Member_Props[i + z2 + 3, 1] = NXPropSet_a_IDs[z]
                z += 3
                z2 += 4    
        
        # collect the property ID of the artificial member into the member property ID array
        if i == len(Member_Props[:,0])-4-5:
            Member_Props[len(Member_Props[:,0])-9:len(Member_Props[:,0]), 0] = np.ones([4 + 5], dtype=int) * int(NXPropSet_a_IDs[len(NXPropSet_a_IDs)-1])
            Member_Props[len(Member_Props[:,0])-9:len(Member_Props[:,0]), 1] = np.ones([4 + 5], dtype=int) * int(NXPropSet_a_IDs[len(NXPropSet_a_IDs)-1])
        
    ### Calculate joint coordinates according to lattice tower specifications and define members
    # calculate original brace coordinates
    brace_coords = get_brace_coords(bee.Nseg, bee.a_Bot, bee.a_Top, bee.b_Bot, bee.b_Top, alpha_x, bee.alpha_y, Tower_Height, tolerance)
    bee.mp_x_top = brace_coords[0]
    bee.mp_y_top = brace_coords[1]
    beta_xp = brace_coords[2]
    beta_xm = brace_coords[3]
    beta_yp = brace_coords[4]
    beta_ym = brace_coords[5]
    brace_coords_xp = brace_coords[6]
    brace_coords_xm = brace_coords[7]
    brace_coords_yp = brace_coords[8]
    brace_coords_ym = brace_coords[9]
    
    # avarage brace coordinates and write joints and member arrays
    Joints_and_Members = get_joints_and_members(bee.Nseg, bee.NJoints, bee.NMembers, NInterf, bee.a_Top, bee.a_Bot, bee.b_Bot, alpha_x, bee.alpha_y, Tower_Height, precone, Overhang, Twr2Shft, NacCMxn, NacCMzn, CM_blade_root, D_tower_top, bee.mp_x_top, bee.mp_y_top, beta_xp, beta_yp, beta_xm, beta_ym, brace_coords_xp, brace_coords_yp, brace_coords_xm, brace_coords_ym, Member_Props, L_Ap)
    bee.joint_coords = Joints_and_Members[0]
    bee.member_a = Joints_and_Members[1]
    bee.base_react_joints = Joints_and_Members[2]
    bee.interface_joints = Joints_and_Members[3]
    
    # add the joint IDs of the interface to the Cmass_a
    # Get additional masses and inertias on tower top
    Cmass_a = Masses_and_Inertias(root_main, bee.a_Top, D_tower_top, precone, Overhang, Twr2Shft, NacCMxn, NacCMzn, mass_Nacelle, mass_hub, mass_blade, CM_blade_root, inertia_nacelle_yaw, inertia_hub_CM)
    Cmass_a[:,0] = bee.interface_joints[5:7]

    #Get_Coordinates.plot_tower(bee.a_Bot, bee.a_Top, bee.b_Bot, bee.b_Top, Tower_Height, beta_xp, beta_yp, beta_xm, beta_ym, mp_x_top, mp_y_top, joint_coords, member_a, brace_coords_xp, brace_coords_yp, brace_coords_xm, brace_coords_ym)

    # Interpolate Tower Parameters
    # get the heights of each leg node
    leg_node_heights = bee.joint_coords[0:(bee.Nseg+1)*4,3]
    mean_node_heights = np.zeros([bee.Nseg+1], dtype = float)
    mean_segment_heights = np.zeros([bee.Nseg], dtype = float)
    
    # calculate the average height of all 4 leg nodes of one segment
    for i in range(len(mean_node_heights)):
        mean_node_heights[i] = (leg_node_heights[i] + leg_node_heights[bee.Nseg+1 + i]+ leg_node_heights[(bee.Nseg+1)*2 + i] + leg_node_heights[(bee.Nseg+1)*3 + i]) / 4    

    # calculate the avarage segment height of each segment
    for i in range(bee.Nseg):
        mean_segment_heights[i] = (mean_node_heights[i] + mean_node_heights[i+1]) / 2

        
    inter_height = np.max(mean_segment_heights) - np.min(mean_segment_heights) # calculate the interpolation height
    
    interpolated_parameters = np.zeros([bee.Nseg * 3 * 4, 4],dtype=float)
    
    # insert interpolated parameter values into respective parameter array for ABAQUS modelling
    count = 0
    for i in range(bee.Nseg):
        count1 = 0
        for ii in range(3):
            count2 = 0
            for iii in range(4):
                interpolated_parameters[count,0] = count
                if count1 == 0: # get parameter values for the legs of segment i
                    if count2 == 0:
                        interpolated_parameters[count,1] = Tower_Parameters[0,1] - (Tower_Parameters[0,1] - Tower_Parameters[6,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = Llmin
                        interpolated_parameters[count,3] = Llmax
                    if count2 == 1:
                        interpolated_parameters[count,1] = Tower_Parameters[0,1] - (Tower_Parameters[0,1] - Tower_Parameters[6,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = Llmin
                        interpolated_parameters[count,3] = Llmax
                    if count2 == 2:
                        interpolated_parameters[count,1] = Tower_Parameters[1,1] - (Tower_Parameters[1,1] - Tower_Parameters[7,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = tlmin
                        interpolated_parameters[count,3] = tlmax
                    if count2 == 3:
                        interpolated_parameters[count,1] = Tower_Parameters[1,1] - (Tower_Parameters[1,1] - Tower_Parameters[7,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = tlmin
                        interpolated_parameters[count,3] = tlmax
                if count1 == 1: # get parameter values for the braces of segment i first a side then b side
                    if count2 == 0:
                        interpolated_parameters[count,1] = Tower_Parameters[2,1] - (Tower_Parameters[2,1] - Tower_Parameters[8,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = Lbmin
                        interpolated_parameters[count,3] = Lbmax
                    if count2 == 1:
                        interpolated_parameters[count,1] = Tower_Parameters[2,1] - (Tower_Parameters[2,1] - Tower_Parameters[8,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = Lbmin
                        interpolated_parameters[count,3] = Lbmax
                    if count2 == 2:
                        interpolated_parameters[count,1] = Tower_Parameters[3,1] - (Tower_Parameters[3,1] - Tower_Parameters[9,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = tbmin
                        interpolated_parameters[count,3] = tbmax
                    if count2 == 3:
                        interpolated_parameters[count,1] = Tower_Parameters[3,1] - (Tower_Parameters[3,1] - Tower_Parameters[9,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = tbmin
                        interpolated_parameters[count,3] = tbmax
                if count1 == 2: # get parameter values for the braces of segment i first a side then b side
                    if count2 == 0:
                        interpolated_parameters[count,1] = Tower_Parameters[4,1] - (Tower_Parameters[4,1] - Tower_Parameters[10,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = Lbmin
                        interpolated_parameters[count,3] = Lbmax
                    if count2 == 1:
                        interpolated_parameters[count,1] = Tower_Parameters[4,1] - (Tower_Parameters[4,1] - Tower_Parameters[10,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = Lbmin
                        interpolated_parameters[count,3] = Lbmax
                    if count2 == 2:
                        interpolated_parameters[count,1] = Tower_Parameters[5,1] - (Tower_Parameters[5,1] - Tower_Parameters[11,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = tbmin
                        interpolated_parameters[count,3] = tbmax
                    if count2 == 3:
                        interpolated_parameters[count,1] = Tower_Parameters[5,1] - (Tower_Parameters[5,1] - Tower_Parameters[11,1]) / inter_height * (mean_segment_heights[i] - np.min(mean_segment_heights))
                        interpolated_parameters[count,2] = tbmin
                        interpolated_parameters[count,3] = tbmax
                count += 1
                count2 += 1
            count1 += 1    
    
    bee.interpolated_parameters = interpolated_parameters
    
    # write ANSYS input file
    data_ansys = ["\n" for x in range(3 * bee.Nseg * 2 + 1)]
    data_ansys[0] = "/PREP7\n"
    z=1
    z2 = 0
    for i in range(1,3 * bee.Nseg * 2 + 1,2):
        data_ansys[i] = "SECTYPE, "+str(z)+", BEAM, L, L-Beam"+str(z)+", 5\n"
        data_ansys[i+1] = "SECDATA, "+str(interpolated_parameters[z2+1,1])+","+str(interpolated_parameters[z2,1])+","+str(interpolated_parameters[z2+3,1])+","+str(interpolated_parameters[z2+2,1])+"\n"
        z2 += 4
        z += 1

    if os.path.exists(root + "ANSYS_Calc\\ansys_calc_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".ans"):
        os.remove(root + "ANSYS_Calc\\ansys_calc_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".ans")
    filename=root + "ANSYS_Calc\\ansys_calc_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".ans"
    with open(filename, 'w') as file:
            file.writelines( data_ansys )
            file.close

    # create batch file for this bee
    ansys_batch = ["\n" for x in range(4)]
    ansys_batch[0] = "@echo off\n"
    ansys_batch[1] = "D:\n"
    ansys_batch[2] = "cd "+root+"ANSYS_Calc\n"
    ansys_batch[3] = "ansys180 -b -i ansys_calc_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".ans -o ansys_calc_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".out"
    filename = root + "ANSYS_Calc\\ANSYS_batch_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".bat"
    with open(filename, 'w') as file:
            file.writelines( ansys_batch )
            file.close
        
    os.system(root + "ANSYS_Calc\\ANSYS_batch_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".bat")
    
    filename= root + "ANSYS_Calc\\ansys_calc_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".out"
    with open(filename, 'r') as file:
            data_raw = file.readlines()

    # get parameters of each parameter set from the ANSYS results
    # Define cross-sections with properties: PropSetID, YoungE, ShearG, MatDens, XsecA, XsecAx, XsecAy, XsecAxy, Xsecxs, Xsecys, XsecJxx, XsecJyy, XsecJxy, XsecJ0
    NXPropSet_a = np.zeros([3 * bee.Nseg + 1, 16], dtype=float)
    NXPropSets = len(NXPropSet_a[:,0])
    z=0
    for i in range(len(data_raw)):
        data_tmp = data_raw[i]
        if data_tmp[0:25] == "   SECTION ID NUMBER IS: ":
            SecID = float(data_tmp[25:])
            data_tmp = data_raw[i+4] # get XsecA
            XsecA = float(data_tmp[26:])
            data_tmp = data_raw[i+5] # get XsecJxx
            XsecJxx = float(data_tmp[26:])
            data_tmp = data_raw[i+6] # get XsecJxy
            XsecJxy = float(data_tmp[26:])
            data_tmp = data_raw[i+7] # get XsecJyy
            XsecJyy = float(data_tmp[26:])
            data_tmp = data_raw[i+9] # get XsecJ0
            XsecJ0 = float(data_tmp[26:])
            data_tmp = data_raw[i+10] # get cx
            cx = float(data_tmp[26:])
            data_tmp = data_raw[i+11] # get cy
            cy = float(data_tmp[26:])
            data_tmp = data_raw[i+12] # get sx
            sx = float(data_tmp[26:])
            data_tmp = data_raw[i+13] # get sy
            sy = float(data_tmp[26:])
            data_tmp = data_raw[i+14] # get kx
            kx = float(data_tmp[26:])
            data_tmp = data_raw[i+15] # get kxy
            kxy = float(data_tmp[26:])
            data_tmp = data_raw[i+16] # get ky
            ky = float(data_tmp[26:])
            
            NXPropSet_a[z,0] = SecID
            NXPropSet_a[z,1] = E
            NXPropSet_a[z,2] = G
            NXPropSet_a[z,3] = rho_steel
            NXPropSet_a[z,4] = XsecA
            NXPropSet_a[z,5] = XsecA * kx
            NXPropSet_a[z,6] = XsecA * ky
            NXPropSet_a[z,7] = XsecA * kxy
            NXPropSet_a[z,8] = -cx + sx
            NXPropSet_a[z,9] = -cy + sy
            NXPropSet_a[z,10] = XsecJxx
            NXPropSet_a[z,11] = XsecJyy
            NXPropSet_a[z,12] = XsecJxy
            NXPropSet_a[z,13] = XsecJ0
            NXPropSet_a[z,14] = cx
            NXPropSet_a[z,15] = cy
    
            z += 1
    
    # add a scaled version of the last NXProp to the NXPropSet_a. The stiffness is scaled up and the mass ist scaled down with factor 1e3
    NXPropSet_a[z,:] = NXPropSet_a[z-1,:]
    NXPropSet_a[z,0] = NXPropSet_a[z,0] + 1
    NXPropSet_a[z,1] = NXPropSet_a[z,1] * 1e3
    NXPropSet_a[z,2] = NXPropSet_a[z,2] * 1e3
    NXPropSet_a[z,3] = NXPropSet_a[z,3] / 1e3
    
    bee.NXPropSet_a = NXPropSet_a
    
    ### Write data to SubDyn-Input
    filename = root_main + "Lattice_Rotatable_raw.txt"
    with open(filename, 'r') as file:
            data_raw = file.readlines()
    
    data = list(data_raw)
    
    data=["\n" for x in range(55 + bee.NJoints + bee.NMembers + NReact + NInterf + NPropSets + NXPropSets + NCmass + NMOutputs + SSoutList_Lines)]
    data[0:17] = data_raw[0:17]
    data[9] = '            '+str(NDiv)+'  NDiv        - Number of sub-elements per member\n'
    data[11] = '            '+str(bee.Nmodes)+'  Nmodes      - Number of internal modes to retain (ignored if CBMod=False). If Nmodes=0 --> Guyan Reduction.\n'
    data[12] = '             '+str(JDampings)+'   JDampings   - Damping Ratios for each retained mode (% of critical) If Nmodes>0, list Nmodes structural damping ratios for each retained mode (% of critical), or a single damping ratio to be applied to all retained modes. (last entered value will be used for all remaining modes).\n'
    data[14] = '             '+str(bee.NJoints)+'   NJoints     - Number of joints (-)\n'
    for i in range(bee.NJoints):
        data[17 + i] = '   ' + str(int(bee.joint_coords[i,0])) + '             ' + str(round(bee.joint_coords[i,1],4)) + '             ' + str(round(bee.joint_coords[i,2],4)) + '             ' + str(round(bee.joint_coords[i,3],4)) + '\n'
    data[17 + bee.NJoints : 17 + bee.NJoints + 4] = data_raw[17:21]
    data[17 + bee.NJoints + 1] = '             '+str(int(NReact))+'   NReact      - Number of Joints with reaction forces; be sure to remove all rigid motion DOFs of the structure  (else det([K])=[0])\n'
    for i in range(NReact):
        data[17 + bee.NJoints + 4 + i] = '    ' + str(int(bee.base_react_joints[i])) + '          1            1            1            1            1            1' + '\n' 
        
    data[17 + bee.NJoints + 4 + NReact : 17 + bee.NJoints + 4 + NReact + 4] = data_raw[21:25]
    data[17 + bee.NJoints + 4 + NReact + 1] = '             '+str(int(NInterf))+'   NInterf     - Number of interface joints locked to the Transition Piece (TP):  be sure to remove all rigid motion dofs\n'
    for i in range(NInterf):
        data[17 + bee.NJoints + 4 + NReact + 4 + i] = '    ' + str(int(bee.interface_joints[i])) + '          1            1            1            1            1            1' + '\n' 
    
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf : 17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4] = data_raw[25:29]
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 1] = '             '+str(int(bee.NMembers))+'   NMembers    - Number of frame members\n'
    for i in range(bee.NMembers):
        data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + i] = '   ' + str(int(bee.member_a[i,0])) + '             ' + str(int(bee.member_a[i,1])) + '             ' + str(int(bee.member_a[i,2])) + '             ' + str(int(bee.member_a[i,3])) + '             ' + str(int(bee.member_a[i,4])) + '             ' + str(int(bee.member_a[i,5])) + '             ' + str(round(bee.member_a[i,6],4)) + '             ' + str(round(bee.member_a[i,7],4)) + '             ' + str(round(bee.member_a[i,8],4)) + '             ' + str(round(bee.member_a[i,9],4)) + '\n'
    
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers: 17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4] = data_raw[29:33]
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 1] = '             '+str(int(NPropSets))+'   NPropSets   - Number of structurally unique x-sections (i.e. how many groups of X-sectional properties are utilized throughout all of the members)\n'
    for i in range(NPropSets):
        data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + i] = ' '
    
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets: 17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4] = data_raw[33:37]
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 1] = '             '+str(int(NXPropSets))+'   NXPropSets  - Number of structurally unique non-circular x-sections (if 0 the following table is ignored)\n'
    for i in range(NXPropSets):
        data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + i] = '  ' + str(int(NXPropSet_a[i,0])) + '     ' + str(NXPropSet_a[i,1]) + '     ' + str(NXPropSet_a[i,2]) + '     ' + str(NXPropSet_a[i,3]) + '     ' + str(NXPropSet_a[i,4]) + '     ' + str(NXPropSet_a[i,5]) + '     ' + str(NXPropSet_a[i,6]) + '     ' + str(NXPropSet_a[i,7]) + '     ' + str(NXPropSet_a[i,8]) + '     ' + str(NXPropSet_a[i,9]) + '     ' + str(NXPropSet_a[i,10]) + '     ' + str(NXPropSet_a[i,11]) + '     ' + str(NXPropSet_a[i,12]) + '     ' + str(NXPropSet_a[i,13]) + '\n'
    
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets: 17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4] = data_raw[37:41]
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 1] = '             '+str(int(NCmass))+'   NCmass      - Number of joints with concentrated masses; Global Coordinate System\n'
    for i in range(NCmass):
        data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4 + i] = str(Cmass_a[i,0])+'    ' + str(Cmass_a[i,1]) + '    '+str(Cmass_a[i,2])+'    '+str(Cmass_a[i,3])+ '    '+str(Cmass_a[i,4])+'\n'
    
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4 + NCmass: 17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4 + NCmass + 13] = data_raw[41:54]
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4 + NCmass + 10] = '             '+str(int(NMOutputs))+'   NMOutputs   - Number of members whose forces/displacements/velocities/accelerations will be output (-) [Must be <= 9].\n'
    for i in range(NMOutputs):
        data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4 + NCmass + 13 + i] = MOutputs_a[i,0] + '\n'
    
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4 + NCmass + 13 + NMOutputs] = data_raw[54]
    for i in range(len(Out_Param_a)):
        data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4 + NCmass + 13 + NMOutputs + 1 + i] = Out_Param_a[i,0] + '\n'
    
    data[17 + bee.NJoints + 4 + NReact + 4 + NInterf + 4 + bee.NMembers + 4 + NPropSets + 4 + NXPropSets + 4 + NCmass + 13 + NMOutputs + 1 + len(Out_Param_a)] = data_raw[63]
    
    filename = root + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".txt"
    with open(filename, 'w') as file:
            file.writelines( data )
            file.close    
        
    # Write batch files for this bee
    # SubDyn driver file for this bee
    filename = root_main + "Lattice_Rotatable_raw.dvr"
    with open(filename, 'r') as file:
            data_raw = file.readlines()
    data = list(data_raw)
    data[4] = str(g)+"               Gravity        - Gravity (m/s^2)\n"
    data[7] = '"IRLT_cycle' + str(cycle) + '_bee'+ str(bee_ID) + '.txt"        SDInputFile\n'
    data[8] = '"' + root_out + 'IRLT_cycle' + str(cycle) + '_bee'+ str(bee_ID) + '"        OutRootName\n'
    data[9] = str(NSteps1) + "                NSteps         - Number of time steps in the simulations (-)\n"
    data[10] = str(TimeInterval1) + "              TimeInterval   - TimeInterval for the simulation (sec)\n"
    data[11] = str(bee.mp_x_top - bee.a_Top/2 + D_tower_top/2) + " " + str(bee.mp_y_top) + " " + str(Tower_Height) + " TP_RefPoint    - Location of the TP reference point in global coordinates (m)\n"
    data[12] = "0.0                SubRotateZ     - Rotation angle of the structure geometry in degrees about the global Z axis.\n"
    filename=root + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".dvr"
    with open(filename, 'w') as file:
            file.writelines( data )
            file.close  
    
    # SubDyn input file for this bee
    bee_batch = ["\n" for x in range(5)]
    bee_batch[0] = "@echo off\n"
    bee_batch[1] = "D:\n"
    bee_batch[2] = "cd "+root+"\n"
    bee_batch[3] = ">NUL SDNK IRLT_cycle" + str(cycle) + "_bee"+ str(bee_ID) + ".dvr" +"\n"
    bee_batch[4] = "exit\n"

    filename=root +"BATCH_cycle"+str(cycle)+'_bee'+str(bee_ID)+".bat"
    with open(filename, 'w') as file:
            file.writelines( bee_batch )
            file.close

    # Write dimensions to dim file
    datafile = ["\n" for x in range(int(50e2))]
    datafile[0] = "Bee "+str(bee_ID)+"\n"
    for i in range(len(Tower_Parameters[:,1])):
        datafile[i+1] = str(Tower_Parameters[i,1]) + "\n"
    filename=root_out + "IRLT_cycle"+str(cycle)+"_bee"+str(bee_ID)+"_optimization_parameters.txt"
    with open(filename, 'w') as file:
            file.writelines( datafile )
            file.close
    
    # lunch that batch file
    #os.system(filename)
    return

def prep_obj_func1(bee_ID, cycle, bee):
    test=2
    return test

def obj_func(bee, cycle):
    datafile = ["\n" for x in range(int(5e4))]
    datafile[0] = "Bee "+str(bee.bee_ID)+"\n"
    # Access results
    try:
       
        if np.max(bee.utilization_a[:,:,2]) > 1 and bee.natFreq[0] < f_min:
            obj_value = bee.mass / 347460.0 * 0.01 + np.max(bee.utilization_a[:,:,2])  * 0.495 +  f_min / bee.natFreq[0] * 0.495 + 3
            
        if np.max(bee.utilization_a[:,:,2]) <= 1 and bee.natFreq[0] < f_min:
            obj_value = bee.mass / 347460.0 * 0.03 +  f_min / bee.natFreq[0] * 0.97 + 1
            
        if np.max(bee.utilization_a[:,:,2]) > 1 and bee.natFreq[0] >= f_min:
            obj_value = bee.mass / 347460.0 * 0.03 + np.max(bee.utilization_a[:,:,2])  * 0.97 + 1
            
        if np.max(bee.utilization_a[:,:,2]) <= 1 and bee.natFreq[0] >= f_min:
            obj_value = (bee.mass / 347460.0)**2
        
        print "Bee " + str(bee.bee_ID) + " mass: " + fs(bee.mass,6) + " f1: " + fs(bee.natFreq[0],6) + " Nseg: " + fs(bee.Nseg,3) + " alpha_y: " + fs(bee.alpha_y,6) + " sF: " + fs(bee.sF,3) + " util_max: " + fs(np.max(bee.utilization_a[:,:,2]),6) + " obj_value: " + fs(obj_value, 6)
        
        datafile[1] = fs(obj_value,6) + " : obj_value\n"
        datafile[2] = fs(bee.mass,6) +  " : mass [kg]\n"
        datafile[3] = fs(bee.natFreq[0],6) + " : 1. Eigenfrequency [Hz]\n"
        datafile[4] = fs(bee.alpha_y,6) + ": alpha_y\n"
        datafile[5] = fs(bee.Nseg,6) + ": Nseg\n"
        datafile[6] = fs(fyk, 6) + ": fyk\n"
        datafile[7] = fs(gamma_m, 6) + ": gamma_m\n"
        datafile[8] = fs(fyk/gamma_m, 6) + ": fyk/gamma_m\n"
        datafile[9] = fs(buckle_min, 6) + ": mininum buckle factor\n"
        datafile[10] = fs(E, 6) + ": E\n"
        datafile[11] = fs(G, 6) + ": G\n"
        datafile[12] = fs(nu, 6) + ": nu\n"
        datafile[13] = fs(rho_steel, 6) + ": rho_steel\n"
        datafile[14] = fs(n_employed_bees, 6) + ": n_employed_bees\n"
        datafile[15] = fs(n_onlooker_bees, 6) + ": n_onlooker_bees\n"
        datafile[16] = fs(n_scout_bees, 6) + ": n_scout_bees\n"
        datafile[17] = fs(Tower_Height, 6) + ": Tower_Height\n"
        datafile[18] = fs(bee.a_Top, 6) + ": a_Top\n"
        datafile[19] = fs(bee.b_Top, 6) + ": b_Top\n"
        datafile[20] = fs(bee.a_Bot, 6) + ": a_Bot\n"
        datafile[21] = fs(bee.b_Bot, 6) + ": b_Bot\n"
        datafile[22] = fs(Nseg_min, 6) + ": Nseg_min\n"
        datafile[23] = fs(Nseg_max, 6) + ": Nseg_max\n"
        datafile[24] = fs(alpha_y_min, 6) + ": alpha_y_min\n"
        datafile[25] = fs(alpha_y_max, 6) + ": alpha_y_max\n"
        datafile[26] = fs(alpha_x*180/np.pi, 6) + ": alpha_x / deg \n"
        datafile[27] = fs(bee.alpha_y*180/np.pi, 6) + ": alpha_y / deg \n"
        datafile[28] = fs(bee.sF_max, 6) + ": sF_max\n"
        datafile[29] = fs(bee.sF_min, 6) + ": sF_min\n"
        datafile[30] = fs(bee.sF, 6) + ": sF\n"
        datafile[31] = fs(0, 6) + ": 0\n"
        datafile[32] = fs(0, 6) + ": 0\n"
        datafile[33] = fs(0, 6) + ": 0\n"
        datafile[34] = fs(0, 6) + ": 0\n"
        datafile[35] = fs(Llmax, 6) + ": Llmax (leg)\n"
        datafile[36] = fs(Llmin, 6) + ": Llmin (leg)\n"
        datafile[37] = fs(tlmax, 6) + ": tlmax (leg)\n"
        datafile[38] = fs(tlmin, 6) + ": tlmin (leg)\n"
        datafile[39] = fs(Lbmax, 6) + ": Lbmax (brace)\n"
        datafile[40] = fs(Lbmin, 6) + ": Lbmin (brace)\n"
        datafile[41] = fs(tbmax, 6) + ": tbmax (brace)\n"
        datafile[42] = fs(tbmin, 6) + ": tbmin (brace)\n"
        datafile[43] = fs(f_min, 6) + ": f min required\n"
        for ii in range(n_DLC):
           datafile[44 + ii] = fs(np.max(bee.utilization_a[ii,:,0]), 6) + ": DLC" + str(ii + 1) + " maximum of buckling utilization of each member\n" 
        for ii in range(n_DLC):
           datafile[44 + n_DLC + ii] = fs(np.max(bee.utilization_a[ii,:,1]), 6) + ": DLC" + str(ii + 1) + " maximum of strength utilization of each member\n"         
        for ii in range(n_DLC):
           datafile[44 + 2 * n_DLC + ii] = fs(np.max(bee.utilization_a[ii,:,2]), 6) + ": DLC" + str(ii + 1) + " maximum of maxiumum utilization of each member\n"            
        datafile[44 + 3 * n_DLC] = fs(np.max(bee.utilization_a[:,:,0]), 6) + ": DLC all: maximum of buckling utilization of each member\n"
        datafile[44 + 3 * n_DLC + 1] = fs(np.max(bee.utilization_a[:,:,1]), 6) + ": DLC all:  maximum of strength utilization of each member\n" 
        datafile[44 + 3 * n_DLC + 2] = fs(np.max(bee.utilization_a[:,:,2]), 6) + ": DLC all maximum of maxiumum utilization of each member\n" 
        for ii in range(n_DLC):
           datafile[44 + 3 * n_DLC + 2 + ii] = fs(np.mean(bee.utilization_a[ii,:,0]), 6) + ": DLC" + str(ii + 1) + " mean of buckling utilization of each member\n" 
        for ii in range(n_DLC):
           datafile[44 + 4 * n_DLC + 2 + ii] = fs(np.mean(bee.utilization_a[ii,:,1]), 6) + ": DLC" + str(ii + 1) + " mean of strength utilization of each member\n"         
        for ii in range(n_DLC):
           datafile[44 + 5 * n_DLC + 2 + ii] = fs(np.mean(bee.utilization_a[ii,:,2]), 6) + ": DLC" + str(ii + 1) + " mean of maxiumum utilization of each member\n"            
        datafile[44 + 6 * n_DLC + 2] = fs(np.mean(bee.utilization_a[:,:,0]), 6) + ": DLC all: mean of buckling utilization of each member\n"
        datafile[44 + 6 * n_DLC + 3] = fs(np.mean(bee.utilization_a[:,:,1]), 6) + ": DLC all:  mean of strength utilization of each member\n" 
        datafile[44 + 6 * n_DLC + 4] = fs(np.mean(bee.utilization_a[:,:,2]), 6) + ": DLC all mean of maxiumum utilization of each member\n"         
        datafile[44 + 6 * n_DLC + 5] = "### Member utilization in each DLC ###\n"    
        for dlci in range(n_DLC):        
            for ii in range(len(bee.utilization_a[dlci,:,2])):
                datafile[44 + 6 * n_DLC + 6 + ii + dlci * len(bee.utilization_a[dlci,:,2])] = "DLC " + fs(dlci+1,2) + " : M " + fs(ii+1,3) +"   "+fs(bee.utilization_a[dlci,ii,0],6)+"   "+fs(bee.utilization_a[dlci,ii,1],6) + " : DLC ; Member ; Buckle_Util ; Stress_Util\n"
        
        filename=root_out+"/IRLT_cycle"+str(cycle)+"_bee"+str(bee.bee_ID)+"_results.txt"
        with open(filename, 'w') as file:
                file.writelines( datafile )
                file.close
    except:
        print "this was unsuccessful:"
        print "Unexpected error:", sys.exc_info()[0]
        print "Bee " + str(bee.bee_ID), " mass: " + fs(bee.mass,6) + " nataural frequency: " + fs(bee.natFreq[0],6) + " Nseg: " + fs(bee.Nseg,3) + " alpha_y: " + fs(bee.alpha_y,6) + " util_max: " + fs(np.max(bee.utilization_a[:,:,2]),6) + " util_mean: " + fs(np.mean(bee.utilization_a[:,:,2]),6)
        obj_value = 1e7
    
    return obj_value

class class_bee:
    def __init__(self, bee_ID, bee_type, Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot):
        self.bee_type = bee_type
        self.bee_ID = bee_ID
        self.parameter_set_inst = parameter_set(Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max)
        self.parameter_set_inst_new = parameter_set(Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max)
        self.fitness = 0
        self.fitness_new = 0
        self.obj_func_value = 1e7
        self.obj_func_value_new = 1e7
        self.local_cycle = 0
        self.local_cycle_max = 50
        self.interpolated_parameters = 0
        self.mass = 0
        self.Nseg = 0
        self.alpha_y = 0
        self.a_Top = a_Top
        self.b_Top = 0
        self.a_Bot = a_Bot
        self.b_Bot = 0
        self.sF = 0
        self.sF_min = sF_min
        self.sF_max = sF_max
        self.NJoints = 0
        self.NMembers = 0
        self.NDiv = 0
        self.Nmodes = 0
        self.mp_x_top = 0
        self.mp_y_top = 0
        self.joint_coords = 0
        self.member_a = 0
        self.NXPropSet_a = 0
        self.base_react_joints = 0
        self.interface_joints = 0
        self.KBBt = 0
        self.MBBt = 0
        self.u = 0
        self.natFreq = 0
        self.TP_Loads = 0
        self.member_loads_static = 0
        self.member_loads_dynamic = 0
        self.utilization_a = 0
        
        #print self.bee_type, " with ID ", self.bee_ID, " has been created."

    def prep_fitness0(self, cycle):
        if (self.bee_type <> "scout bee" and self.local_cycle > self.local_cycle_max):
            self.parameter_set_inst = parameter_set(self.parameter_set_inst.Llmin, self.parameter_set_inst.Llmax, self.parameter_set_inst.tlmin, self.parameter_set_inst.tlmax, self.parameter_set_inst.Lbmin, self.parameter_set_inst.Lbmax, self.parameter_set_inst.tbmin, self.parameter_set_inst.tbmax, self.parameter_set_inst.Nseg_min, self.parameter_set_inst.Nseg_max, self.parameter_set_inst.alpha_y_min, self.parameter_set_inst.alpha_y_max, self.parameter_set_inst.sF_min, self.parameter_set_inst.sF_max)
            self.parameter_set_inst_new = parameter_set(self.parameter_set_inst.Llmin, self.parameter_set_inst.Llmax, self.parameter_set_inst.tlmin, self.parameter_set_inst.tlmax, self.parameter_set_inst.Lbmin, self.parameter_set_inst.Lbmax, self.parameter_set_inst.tbmin, self.parameter_set_inst.tbmax, self.parameter_set_inst.Nseg_min, self.parameter_set_inst.Nseg_max, self.parameter_set_inst.alpha_y_min, self.parameter_set_inst.alpha_y_max, self.parameter_set_inst.sF_min, self.parameter_set_inst.sF_max)            
            self.local_cycle = 0
        prep_obj_func0(self.parameter_set_inst, self.bee_ID, cycle, self)


    def prep_new_fitness0(self, cycle):
        if (self.bee_type <> "scout bee" and self.local_cycle > self.local_cycle_max)  or (self.bee_type <> "scout bee" and cycle != initial_cycle and self.obj_func_value == 1e7):
            self.parameter_set_inst = parameter_set(self.parameter_set_inst.Llmin, self.parameter_set_inst.Llmax, self.parameter_set_inst.tlmin, self.parameter_set_inst.tlmax, self.parameter_set_inst.Lbmin, self.parameter_set_inst.Lbmax, self.parameter_set_inst.tbmin, self.parameter_set_inst.tbmax, self.parameter_set_inst.Nseg_min, self.parameter_set_inst.Nseg_max, self.parameter_set_inst.alpha_y_min, self.parameter_set_inst.alpha_y_max, self.parameter_set_inst.sF_min, self.parameter_set_inst.sF_max)
            self.parameter_set_inst_new = parameter_set(self.parameter_set_inst.Llmin, self.parameter_set_inst.Llmax, self.parameter_set_inst.tlmin, self.parameter_set_inst.tlmax, self.parameter_set_inst.Lbmin, self.parameter_set_inst.Lbmax, self.parameter_set_inst.tbmin, self.parameter_set_inst.tbmax, self.parameter_set_inst.Nseg_min, self.parameter_set_inst.Nseg_max, self.parameter_set_inst.alpha_y_min, self.parameter_set_inst.alpha_y_max, self.parameter_set_inst.sF_min, self.parameter_set_inst.sF_max)
            self.local_cycle = 0
        prep_obj_func0(self.parameter_set_inst_new, self.bee_ID, cycle, self)

    def prep_fitness1(self,cycle):
        prep_obj_func1(self.bee_ID, cycle, self)

    def calculate_fitness(self, cycle):
        self.obj_func_value = obj_func(self, cycle)
        self.fitness = 1 / self.obj_func_value
        self.local_cycle += 1

    def calculate_new_fitness(self, cycle):
        self.obj_func_value_new = obj_func(self, cycle)
        self.fitness_new = 1 / self.obj_func_value_new
        self.local_cycle += 1
    
    def set_obj_func_value(self,value):
        self.obj_func_value = value

    def set_obj_func_value_new(self,value):
        self.obj_func_value_new = value
    
    def fitness_change(self,new_fitness):
        self.fitness = new_fitness
    
    def print_parameters(self):
        self.parameter_set_inst.print_parameter_set()
    
    def print_fitness(self):
        print "fitness old = : ", self.fitness
        print "fitness new = : ", self.fitness_new

class parameter_set:
    def __init__(self, Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max):
        self.n_parameters = 2 * 3 * 2 + 3 # Bottom and Top specifications (2) * leg, bracing a side and bracing b side specification (3) * length and thickness of both L-beam legs (both are the same) (2) + Nseg + alpha_y + sF
        self.Llmin = Llmin
        self.Llmax = Llmax
        self.tlmin = tlmin
        self.tlmax = tlmax
        self.Lbmin = Lbmin
        self.Lbmax = Lbmax
        self.tbmin = tbmin
        self.tbmax = tbmax
        self.Nseg_min = Nseg_min
        self.Nseg_max = Nseg_max
        self.alpha_y_min = alpha_y_min
        self.alpha_y_max = alpha_y_max
        self.sF_min = sF_min
        self.sF_max = sF_max
        #self.a_Top_min = a_Top_min
        #self.a_Top_max = a_Top_max
        #self.b_Top_min = b_Top_min
        #self.b_Top_max = b_Top_max
        #self.a_Bot_min = a_Bot_min
        #self.a_Bot_max = a_Bot_max
        #self.b_Bot_min = b_Bot_min
        #self.b_Bot_max = b_Bot_max
        
        self.parameters = np.empty([self.n_parameters,4])
        self.count = 0
        for i in range(2):
            self.count1 = 0
            for ii in range(3):
                self.count2 = 0
                for iii in range(2):
                    self.parameters[self.count,0] = self.count
                    if self.count1 < 1: # get parameter values for the legs of segment i
                        if self.count2 == 0:
                            self.parameters[self.count,1] = uniform(self.Llmin,self.Llmax)
                            self.parameters[self.count,2] = self.Llmin
                            self.parameters[self.count,3] = self.Llmax
                        if self.count2 == 1:
                            self.parameters[self.count,1] = uniform(self.tlmin,self.tlmax)
                            self.parameters[self.count,2] = self.tlmin
                            self.parameters[self.count,3] = self.tlmax
                    if self.count1 > 0: # get parameter values for the braces of segment i first a side then b side
                        if self.count2 == 0:
                            self.parameters[self.count,1] = uniform(self.Lbmin,self.Lbmax)
                            self.parameters[self.count,2] = self.Lbmin
                            self.parameters[self.count,3] = self.Lbmax
                        if self.count2 == 1:
                            self.parameters[self.count,1] = uniform(self.tbmin,self.tbmax)
                            self.parameters[self.count,2] = self.tbmin
                            self.parameters[self.count,3] = self.tbmax
                    self.count += 1
                    self.count2 += 1
                self.count1 += 1
        self.parameters[self.count,0] = self.count
        self.parameters[self.count,1] = int(uniform(self.Nseg_min,self.Nseg_max))
        self.parameters[self.count,2] = self.Nseg_min
        self.parameters[self.count,3] = self.Nseg_max
        self.count += 1
        self.parameters[self.count,0] = self.count
        self.parameters[self.count,1] = uniform(self.alpha_y_min,self.alpha_y_max)
        self.parameters[self.count,2] = self.alpha_y_min
        self.parameters[self.count,3] = self.alpha_y_max
        self.count += 1
        self.parameters[self.count,0] = self.count
        self.parameters[self.count,1] = uniform(self.sF_min,self.sF_max)
        self.parameters[self.count,2] = self.sF_min
        self.parameters[self.count,3] = self.sF_max
        self.count += 1
        """
        self.parameters[self.count,0] = self.count
        self.parameters[self.count,1] = uniform(self.a_Top_min,self.a_Top_max)
        self.parameters[self.count,2] = self.a_Top_min
        self.parameters[self.count,3] = self.a_Top_max
        self.count += 1
        self.parameters[self.count,0] = self.count
        self.parameters[self.count,1] = uniform(self.b_Top_min,self.b_Top_max)
        self.parameters[self.count,2] = self.b_Top_min
        self.parameters[self.count,3] = self.b_Top_max
        self.count += 1
        self.parameters[self.count,0] = self.count
        self.parameters[self.count,1] = uniform(self.a_Bot_min,self.a_Bot_max)
        self.parameters[self.count,2] = self.a_Bot_min
        self.parameters[self.count,3] = self.a_Bot_max
        self.count += 1
        self.parameters[self.count,0] = self.count
        self.parameters[self.count,1] = uniform(self.b_Bot_min,self.b_Bot_max)
        self.parameters[self.count,2] = self.b_Bot_min
        self.parameters[self.count,3] = self.b_Bot_max
        """
    
    def parameter_set_change(self,parameters):
        self.parameters = parameters
    
    def parameter_change(self,parameter_ID,value):
        self.parameters[parameter_ID,1] = value
    
    def print_parameter_set(self):
        for i in range(len(self.parameters[:,0])):
            print self.parameters[i,0],".: ",self.parameters[i,1]

def create_bees(n_employed_bees, n_onlooker_bees, n_scout_bees, Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot):
    n_bees = n_employed_bees + n_onlooker_bees + n_scout_bees
    bee_list = np.empty([n_bees,3],dtype = object)
    for i in range (n_bees):
        if i < n_employed_bees:
            bee_list[i,1] = class_bee(i,"employed bee",Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot)
        if i >= n_employed_bees and i < n_employed_bees + n_onlooker_bees:
            bee_list[i,1] = class_bee(i,"onlooker bee",Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot)
        if i >= n_employed_bees + n_onlooker_bees:
            bee_list[i,1] = class_bee(i,"scout bee",Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot)
        bee_list[i,2] = bee_list[i,1].bee_type
        bee_list[i,0] = bee_list[i,1].bee_ID
    return bee_list

def create_bees_initial_conditions(rootD, initial_cycle):
    # this function assumes that the number of employed and onlooker bees is the same and that only one scout bee exists
    # count amount of bees
    bees = 0
    while os.path.isfile(rootD + str(bees) + "_cycle"+ str(initial_cycle) + ".txt"):
        bees += 1
    
    # get tower profile dimensions of each bee - Top and Bottom specifications (2) * leg, bracing a side and bracing b side specification (3) * two length and thicknesses per L-beam (4) + Nseg + alpha_y
    parameter_array = np.zeros([bees, 2 * 3 * 4 + 2],dtype=float)
    dimensions_array = np.zeros([bees, 3 * 4 * Nseg_max + 1],dtype=float)
    for ii in range(bees):
        
        # get Nseg and alpha_y of current bee
        filename_rep = root+"\\Tower_Optimization_Report_Bee" + str(ii) + "_cycle"+ str(initial_cycle) + ".txt"
        with open(filename_rep, 'r') as file:
                data_raw_rep = file.readlines()
        
        count_param = 0
        for j in range(len(data_raw_rep)):
            data_tmp = data_raw_rep[j]
            if data_tmp[0] == '\n':
                break
            count_param += 1
        
        if count_param == 10:
            # alpha_y
            alpha_y_string = data_raw_rep[8]
            for jj in range(len(alpha_y_string)):
                ay_stringi = alpha_y_string[jj]
                if ay_stringi == ":":
                    ayj = jj
                    break
            alpha_y_value = float(alpha_y_string[0:ayj])
            # Nseg
            Nseg_string = data_raw_rep[9]
            for jj in range(len(Nseg_string)):
                Nseg_stringi = Nseg_string[jj]
                if Nseg_stringi == ":":
                    Nsegj = jj
                    break
            Nseg_value = int(Nseg_string[0:Nsegj])
        
        if count_param != 10:
            #alpha_y_value = (alpha_y_min + alpha_y_max) / 2.0
            Nseg_value = int(round((Nseg_min + Nseg_max) / 2.0,0))

        parameter_array[ii, 24] = alpha_y_value
        parameter_array[ii, 25] = Nseg_value
        
        # get the member dimensions
        filename = rootD + str(ii) + "_cycle"+ str(initial_cycle) + ".txt"
        with open(filename, 'r') as file:
                data_raw = file.readlines()
    
        for j in range(len(data_raw)):
            if j > 0:
                data_tmp = data_raw[j]
                if data_tmp[0] == '\n':
                    break
                dim_string = data_tmp
                dimension = float(dim_string[0:11])
                dimensions_array[ii,j-1] = dimension
    
    # write bottom and top dimensions to parameter_array
    for i in range(len(dimensions_array[:,0])):
        for j in range(len(dimensions_array[0,:])):
            if np.abs(dimensions_array[i,j]) < 1e-5:
                jmax = j
                break
        parameter_array[i, 0:12] = dimensions_array[i,0:12]
        parameter_array[i, 12:24] = dimensions_array[i,jmax-12:jmax]
        
    n_bees = len(parameter_array[:,0])
    bee_list = np.empty([n_bees,3],dtype = object)
    print "\nFrom ",n_bees, ", ", n_employed_bees, " are employed, ", n_onlooker_bees, " are onlooker and ", n_bees - n_employed_bees - n_onlooker_bees, " are scouts\n"
    for i in range (n_bees):
        if i < n_employed_bees:
            bee_list[i,1] = class_bee(i,"employed bee", Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y)
        if i >= n_employed_bees and i < n_employed_bees + n_onlooker_bees:
            bee_list[i,1] = class_bee(i,"onlooker bee", Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y)
        if i >= n_employed_bees + n_onlooker_bees and i < n_employed_bees + n_onlooker_bees + n_scout_bees:
            bee_list[i,1] = class_bee(i,"scout bee", Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y)
        parameters_tmp = bee_list[i,1].parameter_set_inst.parameters
        parameters_tmp[:,1] = parameter_array[i,:]
        target_param_set = parameters_tmp
        bee_list[i,1].parameter_set_inst.parameter_set_change(target_param_set)
        bee_list[i,2] = bee_list[i,1].bee_type
        bee_list[i,0] = bee_list[i,1].bee_ID
    return bee_list

def calculate_fitness_all(bee_list,bee_type,kind_of_parameter_set_to_use, cycle):
    # Fitness calculation happens in the following order:
    #   1. calculate cross sectional properties (parallelized)
    #   2. calculate the equivalent stiffness matrices of the towers with a first SubDyn analysis
    #   3. calculate the deflections and rotations for each DLC
    #   4. using the deflections and rotations as input for loads analysis in SubDyn (maybe it is possible to analyze all DLC in one calculation by using a time series file, where each timestep represents a certain DLC. This should be possible, because no accelerations and )
    
    # generate batch file to evaluate all equivalent stiffness matrices simultaneously
    all_bees_batch = ["\n" for x in range(n_employed_bees + n_onlooker_bees + 5)]
    all_bees_batch[0] = "@echo off\n"
    all_bees_batch[1] = "D:\n"    
    all_bees_batch[2] = "cd "+root+"\n"
    z = 0
    if kind_of_parameter_set_to_use == "old":
        for x in range(0,len(bee_list[0:,0])):
            if bee_list[x,1].bee_type == bee_type:
                bee_list[x,1].prep_fitness0(cycle)
                all_bees_batch[3 + z] = "start /B Call BATCH_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".bat\n"
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum"):
                    os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum")
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out"):
                    os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out")
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.sum"):
                    os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.sum")
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out"):
                    os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out")
                z += 1
    if kind_of_parameter_set_to_use == "new":
        for x in range(0,len(bee_list[0:,0])):
            if bee_list[x,1].bee_type == bee_type:
                bee_list[x,1].prep_new_fitness0(cycle)
                all_bees_batch[3 + z] = "start /B Call BATCH_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".bat\n"
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum"):
                    os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum")
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out"):
                    os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out")
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.sum"):
                    os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.sum")
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out"):
                    os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out")
                z += 1
    all_bees_batch[3 + z] = "exit"
    
    filename=root +"ALL_Bees_BATCH.bat"
    with open(filename, 'w') as file:
            file.writelines( all_bees_batch )
            file.close
    
    os.system(root + "ALL_Bees_BATCH.bat")
    
    # extract all equivalent stiffness matrices, equivalent mass matrices, the first 20 natural frequencies and masses of each tower
    for x in range(0,len(bee_list[0:,0])):
        gotKBBt, gotMBBt, gotnatFreq, gotmass = False, False, False, False
        if bee_list[x,1].bee_type == bee_type:
            while not gotKBBt and not gotMBBt and not gotnatFreq and not gotmass:
                gotKBBt, gotMBBt, gotnatFreq, gotmass = False, False, False, False
                if not os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum"):
                    time.sleep(0.3)
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum"):
                    time.sleep(0.1)
                    filename = root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum"
                    with open(filename, 'r') as file:
                            data_raw = file.readlines()
                            file.close
                    for i in range(len(data_raw)):
                        data_tmp = data_raw[i]
                        if data_tmp[0:4] == 'KBBt':
                            KBBt_raw = data_raw[i+2 : i+8]
                            bee_list[x,1].KBBt = np.zeros([6,6], dtype=float)
                            gotKBBt = True
                            for j in range(6):
                                KBBt_raw_tmp = KBBt_raw[j]
                                bee_list[x,1].KBBt[j,0] = KBBt_raw_tmp[17:31]
                                bee_list[x,1].KBBt[j,1] = KBBt_raw_tmp[31:46]
                                bee_list[x,1].KBBt[j,2] = KBBt_raw_tmp[46:61]
                                bee_list[x,1].KBBt[j,3] = KBBt_raw_tmp[61:76]
                                bee_list[x,1].KBBt[j,4] = KBBt_raw_tmp[76:91]
                                bee_list[x,1].KBBt[j,5] = KBBt_raw_tmp[91:106]

                        if data_tmp[0:4] == 'MBBt':
                            MBBt_raw = data_raw[i+2 : i+8]
                            bee_list[x,1].MBBt = np.zeros([6,6], dtype=float)
                            gotMBBt = True
                            for j in range(6):
                                MBBt_raw_tmp = MBBt_raw[j]
                                bee_list[x,1].MBBt[j,0] = MBBt_raw_tmp[17:31]
                                bee_list[x,1].MBBt[j,1] = MBBt_raw_tmp[31:46]
                                bee_list[x,1].MBBt[j,2] = MBBt_raw_tmp[46:61]
                                bee_list[x,1].MBBt[j,3] = MBBt_raw_tmp[61:76]
                                bee_list[x,1].MBBt[j,4] = MBBt_raw_tmp[76:91]
                                bee_list[x,1].MBBt[j,5] = MBBt_raw_tmp[91:106]
                            

                        if data_tmp[0:20] == 'FEM Eigenvalues [Hz]':
                            natFreq_raw = data_raw[i+1 : i+21]
                            bee_list[x,1].natFreq = np.zeros([20], dtype=float)
                            gotnatFreq = True
                            for j in range(20):
                                natFreq_raw_tmp = natFreq_raw[j]
                                bee_list[x,1].natFreq[j] = natFreq_raw_tmp[8:21]
                                                            
                        if data_tmp[0:52] == "SubDyn's Total Mass (structural and non-structural)=":
                            mass_raw = data_raw[i]
                            bee_list[x,1].mass = float(mass_raw[54:67]) - mass_Rot - mass_Nacelle
                            gotmass = True
                        
                    if np.size(bee_list[x,1].KBBt) == 1:
                        gotKBBt = False
                    if np.size(bee_list[x,1].MBBt) == 1:
                        gotMBBt = False
                    if np.size(bee_list[x,1].natFreq) == 1:
                        gotnatFreq = False
                    if bee_list[x,1].mass == 0:
                        gotmass = False

    # calculate the deflection and rotation vector of each tower with respect to TP_Load_case and self-weight
    # first calculate the TP_Loads with respect to the TP_Load_case and the self-weight. 
    # The self-weight induces additional loads at the TP, because it is positioned without respect to the self induced displacements.
    # Therefore the first natural frequency analysis can also be used to get the TP_loads without respect to self-weight induced displacements.
    # Furthermore the influence of the RNA self-weight and moments should be canceled out from the TP_loads, because they are included in the TP_load_case loads
     
    for x in range(0,len(bee_list[0:,0])):
        if bee_list[x,1].bee_type == bee_type:
            gotTPloads = False
            while not gotTPloads:
                gotTPloads = False
                if not os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out"):
                    time.sleep(0.2)
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out"):
                    time.sleep(0.1)
                    filename = root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out"
                    with open(filename, 'r') as file:
                        data_raw = file.readlines()
                        file.close()
                    
                    if len(data_raw) == NSteps1 + 8:
                        data_last_row = data_raw[len(data_raw)-1]
                        data_TP_Loads = data_last_row[11 + 6*12 : 11 + 6*12 + 6*12]
                        TP_Loads = np.zeros([6], dtype=float)
                        gotTPloads = True
                        for lc in range(len(TP_Loads)):
                            TP_Loads[lc] = float(data_TP_Loads[lc * 12 : (lc+1)*12])
                        

            # manipulate TP_Loads, so that it represents only tower top deformations due to self-weight of the support structure but not due to the RNA, because RNA loads are included in the TP_load_case laods
            TP_Loads[2] = TP_Loads[2] - mass_Nacelle * g - mass_Rot * g  # subtract the self-weight of RNA
            TP_Loads[4] = TP_Loads[4] + NacCMxn * mass_Nacelle * g - (Overhang + CM_blade_downwind)/2 * mass_Rot * g # subtract the moments from the RNA
            TP_Loads = TP_Loads * (-1) # change sign in order to mimic no loads at TP
            
            bee_list[x,1].u = np.zeros([n_DLC, 6]) # initialize displacements of this bee for each DLC

            # merge TP loads, due to self-weight induced position and the prescribed load cases 
            for dlci in range(n_DLC):
                TP_Load_Cases[dlci,:] = np.copy(TP_Loads[:]) + np.copy(TP_Load_Cases[dlci,:])
                # deformations due to TP loads of DLC dlci
                try:
                    bee_list[x,1].u[dlci,:] = np.dot(np.linalg.inv(bee_list[x,1].KBBt), TP_Load_Cases[dlci,:])
                except:
                    print "bee_ID: ", bee_list[x,1].bee_ID, "\nbee_list[x,1].KBBt: ", bee_list[x,1].KBBt, "\nTP_Load_Cases[dlci,:]: ", TP_Load_Cases[dlci,:]
                    print "ERROR:", sys.exc_info()[0]
                    bee_list[x,1].u[dlci,:] = np.array([1,1,-0.5,0.1,0.1,0.1]) # assumed to be a very hard load case, in order to sort out this tower
    
            # create time series input file for SD driver to account for the different load cases at different points of time
            time_series = ["\n" for xi in range(NSteps2 + n_DLC + 1)]
            dlci = 0            
            for ti in range(NSteps2):
                time_series[ti] = str((ti) * TimeInterval2) + " " + fs(bee_list[x,1].u[dlci,0],7)+ " " + fs(bee_list[x,1].u[dlci,1],7)+ " " + fs(bee_list[x,1].u[dlci,2],7)+ " " + fs(bee_list[x,1].u[dlci,3],7)+ " " + fs(bee_list[x,1].u[dlci,4],7)+ " " + fs(bee_list[x,1].u[dlci,5],7) + 12*" 0.0" + "\n"
            for dlci in range(1,n_DLC):
                time_series[NSteps2 + dlci - 1] = str(NSteps2 * TimeInterval2 + (dlci-1) * TimeInterval2) + " " + fs(bee_list[x,1].u[dlci,0],7)+ " " + fs(bee_list[x,1].u[dlci,1],7)+ " " + fs(bee_list[x,1].u[dlci,2],7)+ " " + fs(bee_list[x,1].u[dlci,3],7)+ " " + fs(bee_list[x,1].u[dlci,4],7)+ " " + fs(bee_list[x,1].u[dlci,5],7) + 12*" 0.0" + "\n"

            filename_time_series = root + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_time_series.txt"
            with open(filename_time_series, 'w') as file:
                file.writelines(time_series)
                file.close()
    
    # manupulate driver input file and tower input file for calculation of each tower with the TP_loads representing displacement and rotation vector u
    for x in range(0,len(bee_list[0:,0])):
       if bee_list[x,1].bee_type == bee_type:
           filename = root + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".dvr"
           with open(filename, 'r') as file:
               data = file.readlines()
               file.close()
           
           data[7] = '"IRLT_cycle' + str(cycle) + '_bee'+ str(bee_list[x,1].bee_ID) + '_static.txt"        SDInputFile\n'
           data[8] = '"' + root_out + 'IRLT_cycle' + str(cycle) + '_bee'+ str(bee_list[x,1].bee_ID) + '_static"        OutRootName\n'
           data[9] = str(NSteps2 + n_DLC - 1) + "              NSteps         - Number of time steps in the simulations (-)\n"
           data[10] = str(TimeInterval2) + "              TimeInterval   - TimeInterval for the simulation (sec)\n"
           data[14] = "   2               InputsMod      - Inputs model {0: all inputs are zero for every timestep, 1: steadystate inputs, 2: read inputs from a file (InputsFile)} (switch)\n"
           data[15] = '"'+root + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_time_series.txt" + '"                 InputsFile     - Name of the inputs file if InputsMod = 2\n'
           #data[17] = str(bee_list[x,1].u[0]) + "   "+str(str(bee_list[x,1].u[1]))+"  "+str(str(bee_list[x,1].u[2]))+"   "+str(str(bee_list[x,1].u[3]))+"   "+str(str(bee_list[x,1].u[4]))+"   "+str(str(bee_list[x,1].u[5]))+"   uTPInSteady     - input displacements and rotations ( m, rads )\n"
           filename = root + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.dvr"           
           with open(filename, 'w') as file:
               file.writelines( data )
               file.close
           
           filename = root + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".txt"
           with open(filename, 'r') as file:
               data = file.readlines()
               file.close
           
           data[12] = '             '+str(JDampings_static)+'   JDampings   - Damping Ratios for each retained mode (% of critical) If Nmodes>0, list Nmodes structural damping ratios for each retained mode (% of critical), or a single damping ratio to be applied to all retained modes. (last entered value will be used for all remaining modes).\n'
           for i in range(len(data)):
               data_tmp = data[i]
               if data_tmp[0:9] == 'CMJointID':     
                   data_tmp2 = data[i + 2]
                   data[i + 2] = data_tmp2[0:6] + "   0.0   0.0   0.0   0.0\n"
                   data_tmp2 = data[i + 3]
                   data[i + 3] = data_tmp2[0:6] + "   0.0   0.0   0.0   0.0\n"
               
               if data_tmp[0:85] == '---------------------------- OUTPUT: SUMMARY & OUTFILE ------------------------------': 
                   data[i + 1] = "False             SSSum       - Output a Summary File (flag).It contains: matrices K,M  and C-B reduced M_BB, M-BM, K_BB, K_MM(OMG^2), PHI_R, PHI_L. It can also contain COSMs if requested.\n"
           filename = root + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.txt"
           with open(filename, 'w') as file:
               file.writelines( data )
               file.close
           
    # generate batch files for static load case
    for x in range(0,len(bee_list[0:,0])):
        bee_ID = bee_list[x,1].bee_ID
        if bee_list[x,1].bee_type == bee_type:
            # SubDyn input file for this bee
            bee_batch = ["\n" for x in range(5)]
            bee_batch[0] = "@echo off\n"
            bee_batch[1] = "D:\n"
            bee_batch[2] = "cd "+root+"\n"
            bee_batch[3] = ">NUL SDNK IRLT_cycle" + str(cycle) + "_bee"+ str(bee_ID) + "_static.dvr\n"
            bee_batch[4] = "exit\n"
            filename=root +"BATCH_cycle"+str(cycle)+'_bee'+str(bee_ID)+"_static.bat"
            with open(filename, 'w') as file:
                file.writelines( bee_batch )
                file.close
        
    all_bees_batch = ["\n" for x in range(n_employed_bees + n_onlooker_bees + 5)]
    all_bees_batch[0] = "@echo off\n"
    all_bees_batch[1] = "D:\n"
    all_bees_batch[2] = "cd "+root+"\n"
    z = 0
    for x in range(0,len(bee_list[0:,0])):
        if bee_list[x,1].bee_type == bee_type:
            all_bees_batch[3 + z] = "start /B Call BATCH_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.bat\n"
            z += 1
    all_bees_batch[3 + z] = "exit\n"
    filename=root +"ALL_Bees_BATCH_static.bat"
    with open(filename, 'w') as file:
        file.writelines( all_bees_batch )
        file.close
    
    # calculate each tower with the TP_loads representing displacement and rotation vector u
    os.system(root + "ALL_Bees_BATCH_static.bat")

    # read each member joint1 and joint2 load components from static load case
    for x in range(0,len(bee_list[0:,0])):
        if bee_list[x,1].bee_type == bee_type:
            bee_list[x,1].member_loads_static = np.zeros([n_DLC, (bee_list[x,1].NMembers)+1, 12])
            bee_list[x,1].member_loads_dynamic = np.zeros([n_DLC, (bee_list[x,1].NMembers)+1, 12])
            gotMemberLoads = False
            while not gotMemberLoads:
                gotMemberLoads = False
                if not os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out"):
                    time.sleep(0.2)
                if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out"):
                    time.sleep(0.3)
                    filename = root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out"
                    with open(filename, 'r') as file:
                        data_raw = file.readlines()
                        file.close()
                    
                    if len(data_raw) == NSteps2 + 8 + (n_DLC - 1):
                        data_last_row = data_raw[len(data_raw)-1]
                        # extract member loads for each DLC
                        for dlci in range(n_DLC):
                            data_last_row = data_raw[len(data_raw)-(n_DLC - dlci)]
                            for nmi in range(0, bee_list[x,1].NMembers):
                                if nmi == 0:
                                    for lci in range(12):
                                        bee_list[x,1].member_loads_static[dlci, nmi,lci] = data_last_row[11 + lci * 12 : 11 + lci * 12 + 12]
                                        bee_list[x,1].member_loads_dynamic[dlci, nmi,lci] = data_last_row[11 + lci * 12 : 11 + lci * 12 + 12]
                                
                                for lci in range(12):
                                    lci_dynamic = lci
                                    lci_dynamic2 = lci
                                    # extract static member loads inclusive reaction and interface loads (as first member)
                                    if lci < 6:
                                        bee_list[x,1].member_loads_static[dlci, nmi+1,lci] = data_last_row[11 + 12*12 + nmi * 12 * 24 + lci * 12 : 11 + 12*12 + nmi * 12 * 24 + lci * 12 + 12]
                                    if lci >= 6:
                                        lci = lci + 6
                                        bee_list[x,1].member_loads_static[dlci, nmi+1,lci-6] = data_last_row[11 + 12*12 + nmi * 12 * 24 + lci * 12 : 11 + 12*12 + nmi * 12 * 24 + lci * 12 + 12]
                                    
                                    # extract dynamic member loads inclusive reaction and interface loads (as first member)                                
                                    if lci_dynamic < 6:
                                        lci_dynamic += 6
                                        bee_list[x,1].member_loads_dynamic[dlci, nmi+1,lci_dynamic-6] = data_last_row[11 + 12*12 + nmi * 12 * 24 + lci_dynamic * 12 : 11 + 12*12 + nmi * 12 * 24 + lci_dynamic * 12 + 12]
                                    if lci_dynamic2 >= 6:
                                        lci_dynamic2 = lci_dynamic2 + 12
                                        bee_list[x,1].member_loads_dynamic[dlci, nmi+1,lci_dynamic2-12] = data_last_row[11 + 12*12 + nmi * 12 * 24 + lci_dynamic2 * 12 : 11 + 12*12 + nmi * 12 * 24 + lci_dynamic2 * 12 + 12]
                            
                        gotMemberLoads = True

    """
    # print maximum dynamic loads in order to check, static convergence
    for x in range(0,len(bee_list[0:,0])):
        if bee_list[x,1].bee_type == bee_type:
            print "bee ",bee_list[x,1].bee_ID, " maximum dynamic load: ", np.max(np.abs(bee_list[x,1].member_loads_dynamic[1:,:]))
    """

    # check each member of each bee for buckling and yielding, except the artificial members at tower top (-9) for each DLC
    for x in range(0,len(bee_list[0:,0])):
        if bee_list[x,1].bee_type == bee_type:
            bee_list[x,1].utilization_a = np.zeros([n_DLC, bee_list[x,1].NMembers - 9, 3])
            for dlci in range(n_DLC):
                for mi in range(bee_list[x,1].NMembers - 9):
                    # get member loads
                    j1_loads = bee_list[x,1].member_loads_static[dlci, 1+mi,0:6]
                    j2_loads = bee_list[x,1].member_loads_static[dlci, 1+mi,6:12]
                    
                    # get parameters of this member
                    memberJ1_ID = bee_list[x,1].member_a[mi, 1]
                    memberJ2_ID = bee_list[x,1].member_a[mi, 2]
                    
                    J1_coords_m = bee_list[x,1].joint_coords[np.where(bee_list[x,1].joint_coords[:,0] == memberJ1_ID)[0][0], 1:]
                    J2_coords_m = bee_list[x,1].joint_coords[np.where(bee_list[x,1].joint_coords[:,0] == memberJ2_ID)[0][0], 1:]
                    
                    member_XPropID = bee_list[x,1].member_a[mi, 3]
    
                    member_XProps = bee_list[x,1].NXPropSet_a[np.where(bee_list[x,1].NXPropSet_a[:,0] == member_XPropID)[0][0], :]
              
                    L_m = np.sqrt((J2_coords_m[0] - J1_coords_m[0])**2 + (J2_coords_m[1] - J1_coords_m[1])**2 + (J2_coords_m[2] - J1_coords_m[2])**2) # length of the member
    
                    XsecA_m = member_XProps[4]
                    XsecJxx_m = member_XProps[10]
                    XsecJyy_m = member_XProps[11]
                    XsecJxy_m = member_XProps[12]
                    XsecJ0_m = member_XProps[13]
                    cx_m = member_XProps[14]    # centroid x-coordinate with respect to the origion, which is placed at the intersection of the midline of each L-profile leg
                    cy_m = member_XProps[15]    # centroid y-coordinate with respect to the origion, which is placed at the intersection of the midline of each L-profile leg
                    
                    # find L-beam dimensions
                    Nseg_m = int((member_XPropID-1)/3.0)  # get the segment number of this member
                    part_m = (member_XPropID - 1)%3.0  # get the part of the segment, where the member is e.g. 0 = leg, 1 = a-side, 2 = b-side
                    
                    L_params_m = np.zeros([4], dtype = float)
                    L_params_m = bee_list[x,1].interpolated_parameters[int(Nseg_m * 12 + part_m * 4) : int(Nseg_m * 12 + part_m * 4 + 4),1]
                    lv_m = L_params_m[0]
                    lh_m = L_params_m[1]
                    tv_m = L_params_m[2]
                    th_m = L_params_m[3]
                    
                    #print "member ", mi+1," with Nseg_m ",Nseg_m," and part_m ",part_m," has L dimensions: ",  L_params_m
                    # check for cross sectional class of this member
                    class3_m = False
                    #class3_m = True
                    epsilon = np.sqrt(235 / (fyk/1e6))
                    lambda_1 = np.pi * np.sqrt(E/fyk)
                    if (lv_m-2*tv_m) / tv_m <= 15*epsilon and (lh_m-2*th_m) / th_m <= 15*epsilon and (lv_m + lh_m)/ (2 * tv_m) <= 11.5 * epsilon and (lv_m + lh_m)/ (2 * th_m) <= 11.5 * epsilon:
                        class3_m = True
                    
                    #print "class3: ", class3_m
                    A_use1 = XsecA_m
                    Jxx_use1 = XsecJxx_m
                    Jyy_use1 = XsecJyy_m
                    Jxy_use1 = XsecJxy_m
                    cx_use1 = cx_m
                    cy_use1 = cy_m
                    dMxEd1 = 0
                    dMyEd1 = 0
                    A_use2 = XsecA_m
                    Jxx_use2 = XsecJxx_m
                    Jyy_use2 = XsecJyy_m
                    Jxy_use2 = XsecJxy_m
                    cx_use2 = cx_m
                    cy_use2 = cy_m
                    dMxEd2 = 0
                    dMyEd2 = 0
                    
                    if not class3_m:
                        # calculate the three normal stresses at three points of a L-Beam: vertical top; vertical bot/horizontal left; horizontal right
                        sigma_vt1 = normal_stress(j1_loads[2],j1_loads[3],j1_loads[4],A_use1,Jxx_use1,Jyy_use1,Jxy_use1, -cx_use1 - tv_m/2.0 , lv_m - th_m/2.0 - cy_use1) # normal stress in the vertical top postion of the L-beam
                        sigma_vbhl1 = normal_stress(j1_loads[2],j1_loads[3],j1_loads[4],A_use1,Jxx_use1,Jyy_use1,Jxy_use1, -cx_use1 - tv_m/2.0, - cy_use1 - th_m/2.0)     # normal stress in the vertical bot/horizontal left postion of the L-beam
                        sigma_hr1 = normal_stress(j1_loads[2],j1_loads[3],j1_loads[4],A_use1,Jxx_use1,Jyy_use1,Jxy_use1, lh_m - tv_m/2.0 - cx_use1, -cy_use1 - th_m/2.0)       # normal stress in the horizontal right postion of the L-beam
                        sigma_vt2 = normal_stress(j2_loads[2],j2_loads[3],j2_loads[4],A_use2,Jxx_use2,Jyy_use2,Jxy_use2, -cx_use2 - tv_m/2.0 , lv_m - th_m/2.0 - cy_use2) # normal stress in the vertical top postion of the L-beam
                        sigma_vbhl2 = normal_stress(j2_loads[2],j2_loads[3],j2_loads[4],A_use2,Jxx_use2,Jyy_use2,Jxy_use2, -cx_use2 - tv_m/2.0, - cy_use2 - th_m/2.0)     # normal stress in the vertical bot/horizontal left postion of the L-beam
                        sigma_hr2 = normal_stress(j2_loads[2],j2_loads[3],j2_loads[4],A_use2,Jxx_use2,Jyy_use2,Jxy_use2, lh_m - tv_m/2.0 - cx_use2, -cy_use2 - th_m/2.0)       # normal stress in the horizontal right postion of the L-beam
    
                        # change signs according to EN 1993-1-5 Table 4.2 definition for pressure as being positive
                        sigma_vt1 = sigma_vt1 * (-1)
                        sigma_vbhl1 = sigma_vbhl1 * (-1)
                        sigma_hr1 = sigma_hr1 * (-1)
                        sigma_vt2 = sigma_vt2 * (-1)
                        sigma_vbhl2 = sigma_vbhl2 * (-1)
                        sigma_hr2 = sigma_hr2 * (-1)
                        
                        # vertical L-beam leg joint 1               
                        if sigma_vt1 > sigma_vbhl1:
                            psiv1 = sigma_vbhl1/sigma_vt1
                            if psiv1 >= 1:
                                psiv1 = 1
                            if psiv1 <= -3:
                                psiv1 = -3
                            k_sigma_v1 = 0.57 - 0.21 * psiv1 + 0.07 * psiv1**2
                        if sigma_vt1 <= sigma_vbhl1:
                            psiv1 = sigma_vt1/sigma_vbhl1
                            if psiv1 >= 1:
                                psiv1 = 1
                                k_sigma_v1 = 0.43
                            if psiv1< 1 and psiv1 > 0:
                                k_sigma_v1 = 0.578 / (psiv1 + 0.34)
                            if psiv1 == 0:
                                k_sigma_v1 = 1.7
                            if psiv1 <0 and psiv1 > -1:
                                k_sigma_v1 = 1.7 - 5 * psiv1 + 17.1 * psiv1**2
                            if psiv1 <= -1:
                                psiv1 = -1
                                k_sigma_v1 = 23.8
                        if sigma_vbhl1 <= sigma_hr1:
                            psih1 = sigma_vbhl1/sigma_hr1
                            if psih1 >= 1:
                                psih1 = 1
                            if psih1 <= -3:
                                psih1 = -3
                            k_sigma_h1 = 0.57 - 0.21 * psih1 + 0.07 * psih1**2
                        if sigma_vbhl1 > sigma_hr1:
                            psih1 = sigma_hr1/sigma_vbhl1
                            if psih1 >= 1:
                                psih1 = 1
                                k_sigma_h1 = 0.43
                            if psih1< 1 and psih1 > 0:
                                k_sigma_h1 = 0.578 / (psih1 + 0.34)
                            if psih1 == 0:
                                k_sigma_h1 = 1.7
                            if psih1 <0 and psih1 > -1:
                                k_sigma_h1 = 1.7 - 5 * psih1 + 17.1 * psih1**2
                            if psih1 <= -1:
                                psih1 = -1
                                k_sigma_h1 = 23.8
                        if sigma_vt1 < 0 and sigma_vbhl1 < 0:
                            psiv1 = 1
                            k_sigma_v1 = 23.8
                        if sigma_hr1 < 0 and sigma_vbhl1 < 0:
                            psih1 = 1
                            k_sigma_h1 = 23.8
    
                        # vertical L-beam leg joint 2
                        if sigma_vt2 > sigma_vbhl2:
                            psiv2 = sigma_vbhl2/sigma_vt2
                            if psiv2 >= 1:
                                psiv2 = 1
                            if psiv2 <= -3:
                                psiv2 = -3
                            k_sigma_v2 = 0.57 - 0.21 * psiv2 + 0.07 * psiv2**2
                        if sigma_vt2 <= sigma_vbhl2:
                            psiv2 = sigma_vt2/sigma_vbhl2
                            if psiv2 >= 1:
                                psiv2 = 1
                                k_sigma_v2 = 0.43
                            if psiv2< 1 and psiv2 > 0:
                                k_sigma_v2 = 0.578 / (psiv2 + 0.34)
                            if psiv2 == 0:
                                k_sigma_v2 = 1.7
                            if psiv2 <0 and psiv2 > -1:
                                k_sigma_v2 = 1.7 - 5 * psiv2 + 17.1 * psiv2**2
                            if psiv2 <= -1:
                                psiv2 = -1
                                k_sigma_v2 = 23.8
                        if sigma_vbhl2 <= sigma_hr2:
                            psih2 = sigma_vbhl2/sigma_hr2
                            if psih2 >= 1:
                                psih2 = 1
                            if psih2 <= -3:
                                psih2 = -3
                            k_sigma_h2 = 0.57 - 0.21 * psih2 + 0.07 * psih2**2
                        if sigma_vbhl2 > sigma_hr2:
                            psih2 = sigma_hr2/sigma_vbhl2
                            if psih2 >= 1:
                                psih2 = 1
                                k_sigma_h2 = 0.43
                            if psih2< 1 and psih2 > 0:
                                k_sigma_h2 = 0.578 / (psih2 + 0.34)
                            if psih2 == 0:
                                k_sigma_h2 = 1.7
                            if psih2 <0 and psih2 > -1:
                                k_sigma_h2 = 1.7 - 5 * psih2 + 17.1 * psih2**2
                            if psih2 <= -1:
                                psih2 = -1
                                k_sigma_h2 = 23.8
                        if sigma_vt2 < 0 and sigma_vbhl2 < 0:
                            psiv2 = 1
                            k_sigma_v2 = 23.8
                        if sigma_hr2 < 0 and sigma_vbhl2 < 0:
                            psih2 = 1
                            k_sigma_h2 = 23.8
                        
    
                        # DER ABMINDERUNGSFAKTOR rho bezieht sich nur auf den angeschlossenen Winkel! Dies ist noch nicht implementiert!
                        b_bar_v = (lv_m - 2*th_m) # according to DIN EN 1993-3-1_2010 6.3.1 (2) a
                        b_bar_h = (lh_m - 2*tv_m) # according to DIN EN 1993-3-1_2010 6.3.1 (2) a
                        
                        lambda_bar_pv1 = b_bar_v / tv_m / (28.4 * epsilon * np.sqrt(k_sigma_v1))
                        lambda_bar_ph1 = b_bar_h / th_m / (28.4 * epsilon * np.sqrt(k_sigma_h1))
                        lambda_bar_pv2 = b_bar_v / tv_m / (28.4 * epsilon * np.sqrt(k_sigma_v2))
                        lambda_bar_ph2 = b_bar_h / th_m / (28.4 * epsilon * np.sqrt(k_sigma_h2))
                        
                        if lambda_bar_pv1 <= 0.748 or (sigma_vt1 < 0 and sigma_vbhl1 < 0):
                            rhov1_eff = 1
                        if lambda_bar_ph1 <= 0.748 or (sigma_vbhl1 < 0 and sigma_hr1 < 0):
                            rhoh1_eff = 1
                        if lambda_bar_pv2 <= 0.748 or (sigma_vt2 < 0 and sigma_vbhl2 < 0):
                            rhov2_eff = 1
                        if lambda_bar_ph2 <= 0.748 or (sigma_vbhl2 < 0 and sigma_hr2 < 0):
                            rhoh2_eff = 1
                        
                        if lambda_bar_pv1 > 0.748:
                            rhov1_eff = (lambda_bar_pv1 - 0.188 ) / lambda_bar_pv1**2
                            if rhov1_eff > 1:
                                rhov1_eff = 1
                        if lambda_bar_ph1 > 0.748:
                            rhoh1_eff = (lambda_bar_ph1 - 0.188 ) / lambda_bar_ph1**2
                            if rhoh1_eff > 1:
                                rhoh1_eff = 1
                        if lambda_bar_pv2 > 0.748:
                            rhov2_eff = (lambda_bar_pv2 - 0.188 ) / lambda_bar_pv2**2
                            if rhov2_eff > 1:
                                rhov2_eff = 1
                        if lambda_bar_ph2 > 0.748:
                            rhoh2_eff = (lambda_bar_ph2 - 0.188 ) / lambda_bar_ph2**2
                            if rhoh2_eff > 1:
                                rhoh2_eff = 1
                        
                        Xsec_eff_values1 = effective_cross_section(psiv1, psih1, rhov1_eff, rhoh1_eff, lv_m, lh_m, tv_m, th_m, sigma_vt1, sigma_vbhl1, sigma_hr1)
                        Xsec_eff_values2 = effective_cross_section(psiv2, psih2, rhov2_eff, rhoh2_eff, lv_m, lh_m, tv_m, th_m, sigma_vt2, sigma_vbhl2, sigma_hr2)
                        
                        #print "A1: ", round(A_use1,6), "Jxx1: ", round(Jxx_use1,6), "Jyy1: ", round(Jyy_use1,6), "Jxy1: ", round(Jxy_use1,6), "A2: ", round(A_use2,6), "Jxx2: ", round(Jxx_use2,6), "Jyy2: ", round(Jyy_use2,6), "Jxy2: ", round(Jxy_use2,6)
                        #print mi," convergence: ",round(np.abs(Xsec_eff_values1[0] - A_use1),6), round(np.abs(Xsec_eff_values1[3] - Jxx_use1),6), round(np.abs(Xsec_eff_values1[4] - Jyy_use1),6), round(np.abs(Xsec_eff_values1[5] - Jxy_use1),6), round(np.abs(Xsec_eff_values2[0] - A_use2),6), round(np.abs(Xsec_eff_values2[3] - Jxx_use2),6), round(np.abs(Xsec_eff_values2[4] - Jyy_use2),6), round(np.abs(Xsec_eff_values2[5] - Jxy_use2),6),  "\n\n"
                        #print "A_use1_old: ", A_use1, "\tJxx_use1_old: ", Jxx_use1, "\tJyy_use1_old: ", Jyy_use1, "\tJxy_use1_old: ", Jxy_use1
                        #print "A_use2_old: ", A_use2, "\tJxx_use2_old: ", Jxx_use2, "\tJyy_use2_old: ", Jyy_use2, "\tJxy_use2_old: ", Jxy_use2
                        A_use1 = Xsec_eff_values1[0]
                        Jxx_use1 = Xsec_eff_values1[3]
                        Jyy_use1 = Xsec_eff_values1[4]
                        Jxy_use1 = Xsec_eff_values1[5]
                        dMxEd1 = (Xsec_eff_values1[2] - cy_use1) * j1_loads[2]
                        dMyEd1 = (cx_use1 - Xsec_eff_values1[1]) * j1_loads[2]
                        cx_use1 = Xsec_eff_values1[1]
                        cy_use1 = Xsec_eff_values1[2]
                        
                        A_use2 = Xsec_eff_values2[0]
                        Jxx_use2 = Xsec_eff_values2[3]
                        Jyy_use2 = Xsec_eff_values2[4]
                        Jxy_use2 = Xsec_eff_values2[5]
                        dMxEd2 = (Xsec_eff_values2[2] - cy_use2) * j2_loads[2]
                        dMyEd2 = (cx_use2 - Xsec_eff_values2[1]) * j2_loads[2]
                        cx_use2 = Xsec_eff_values2[1]
                        cy_use2 = Xsec_eff_values2[2]
                        #print "A_use1: ", A_use1, "\tJxx_use1: ", Jxx_use1, "\tJyy_use1: ", Jyy_use1, "\tJxy_use1: ", Jxy_use1
                        #print "A_use2: ", A_use2, "\tJxx_use2: ", Jxx_use2, "\tJyy_use2: ", Jyy_use2, "\tJxy_use2: ", Jxy_use2
    
    
                    #print "m: "+ fs(mi,4) + "A_old: " + fs(XsecA_m,6) + "XsecJxx_m: " + fs(XsecJxx_m,6) + "XsecJyy_m: " + fs(XsecJyy_m,6) + "XsecJxy_m: " + fs(XsecJxy_m,6) + "cx_m: " + fs(cx_m,4) + "cy_m: " + fs(cy_m,4)
                    #print "A1: " + fs(A_use1,6)+ "Jxx1: "+ fs(Jxx_use1,6)+ "Jyy1: "+ fs(Jyy_use1,6)+ "Jxy1: "+ fs(Jxy_use1,6)+ "A2: "+ fs(A_use2,6)+ " Jxx2: "+ fs(Jxx_use2,6)+ " Jyy2: "+ fs(Jyy_use2,6)+ " Jxy2: "+ fs(Jxy_use2,6) + "cx1: " + fs(cx_use1,4) + "cy1: " + fs(cy_use1,4)+ "cx2: " + fs(cx_use2,4) + "cy2: " + fs(cy_use2,4) + " b_effv1: " + fs(Xsec_eff_values1[6],4) + " b_effh1: " + fs(Xsec_eff_values1[7],4) + " b_effv2: " + fs(Xsec_eff_values2[6],4) + " b_effh2: " + fs(Xsec_eff_values2[7],4)
                    # transform cross sectional values to weak and strong axes v-v and u-u
                    # find minimum second area moment of inertia of this member
                    Juu_use1 = (Jxx_use1 + Jyy_use1) / 2 + np.sqrt((Jxx_use1 - Jyy_use1)**2 / 4 + Jxy_use1**2)
                    Jvv_use1 = (Jxx_use1 + Jyy_use1) / 2 - np.sqrt((Jxx_use1 - Jyy_use1)**2 / 4 + Jxy_use1**2)
                    Juu_use2 = (Jxx_use2 + Jyy_use2) / 2 + np.sqrt((Jxx_use2 - Jyy_use2)**2 / 4 + Jxy_use2**2)
                    Jvv_use2 = (Jxx_use2 + Jyy_use2) / 2 - np.sqrt((Jxx_use2 - Jyy_use2)**2 / 4 + Jxy_use2**2)
                    # calculate the rotation angle of this principal bending axes w.r.t. the initial coordinate system (x-x, y-y)                
                    if not (Jyy_use1-Jxx_use1) == 0:
                        Theta_vv1 = np.arctan(2 * Jxy_use1 / (Jyy_use1-Jxx_use1)) / 2.0 # rotation angle of to the main bending axes
                    if (Jyy_use1-Jxx_use1) == 0:
                        Theta_vv1 = np.sign(Jxy_use1) * 45 * np.pi/180
                    if not (Jyy_use2-Jxx_use2) == 0:
                        Theta_vv2 = np.arctan(2 * Jxy_use2 / (Jyy_use2-Jxx_use2)) / 2.0 # rotation angle of the main bending axes
                    if (Jyy_use2-Jxx_use2) == 0:
                        Theta_vv2 = np.sign(Jxy_use2) * 45 * np.pi/180
                        
                    # calculate the bending moments around the strong and weak axes for both joints
                    M1_uu = (j1_loads[4]+dMyEd1) * np.sin(Theta_vv1) + (j1_loads[3] + dMxEd1) * np.cos(Theta_vv1)
                    M1_vv = (j1_loads[4]+dMyEd1) * np.cos(Theta_vv1) - (j1_loads[3] + dMxEd1) * np.sin(Theta_vv1)
                    M2_uu = (j2_loads[4]+dMyEd2) * np.sin(Theta_vv2) + (j2_loads[3] + dMxEd2) * np.cos(Theta_vv2)
                    M2_vv = (j2_loads[4]+dMyEd2) * np.cos(Theta_vv2) - (j2_loads[3] + dMxEd2) * np.sin(Theta_vv2)
                    
                    # global member buckling resistance
                    i_uu1 = np.sqrt(Juu_use1 / A_use1)
                    i_vv1 = np.sqrt(Jvv_use1 / A_use1)
                    i_uu2 = np.sqrt(Juu_use2 / A_use2)
                    i_vv2 = np.sqrt(Jvv_use2 / A_use2)
                    #i_vv = np.sqrt(J_min_use / A_use)
                    lambda_bar_uu1 = L_m * beta_buckling / (i_uu1 * lambda_1)
                    lambda_bar_vv1 = L_m * beta_buckling / (i_vv1 * lambda_1)
                    lambda_bar_uu2 = L_m * beta_buckling / (i_uu2 * lambda_1)
                    lambda_bar_vv2 = L_m * beta_buckling / (i_vv2 * lambda_1)
                    #lambda_bar_vv = L_m * beta_buckling / (i_vv * lambda_1)
                    Phi_uu1 = 0.5 * (1 + alpha_b * (lambda_bar_uu1 - 0.2) + lambda_bar_uu1**2)
                    Phi_vv1 = 0.5 * (1 + alpha_b * (lambda_bar_vv1 - 0.2) + lambda_bar_vv1**2)
                    Phi_uu2 = 0.5 * (1 + alpha_b * (lambda_bar_uu2 - 0.2) + lambda_bar_uu2**2)
                    Phi_vv2 = 0.5 * (1 + alpha_b * (lambda_bar_vv2 - 0.2) + lambda_bar_vv2**2)
                    #Phi_vv = 0.5 * (1 + alpha_b * (lambda_bar_vv - 0.2) + lambda_bar_vv**2)
                    Chi_uu1 = 1 / (Phi_uu1 + np.sqrt(Phi_uu1**2-lambda_bar_uu1**2))
                    Chi_vv1 = 1 / (Phi_vv1 + np.sqrt(Phi_vv1**2-lambda_bar_vv1**2))
                    Chi_uu2 = 1 / (Phi_uu2 + np.sqrt(Phi_uu2**2-lambda_bar_uu2**2))
                    Chi_vv2 = 1 / (Phi_vv2 + np.sqrt(Phi_vv2**2-lambda_bar_vv2**2))
                    #Chi_vv = 1 / (Phi_vv + np.sqrt(Phi_vv**2-lambda_bar_vv**2))
                    if Chi_uu1 > 1:
                        Chi_uu1 = 1
                    if Chi_vv1 > 1:
                        Chi_vv1 = 1
                    if Chi_uu2 > 1:
                        Chi_uu2 = 1
                    if Chi_vv2 > 1:
                        Chi_vv2 = 1
                    #if Chi_vv > 1:
                        #Chi_vv = 1
                    #NbRd_xx = Chi_xx * A_use * fyk / gamma_buckle
                    #NbRd_vv = Chi_vv * A_use * fyk / gamma_buckle
                    #NbRd_vv = Chi_vv * A_use * fyk / gamma_buckle
                    
                    # global member torsional buckling
                    # corner coordinates of L-beam w.r.t. centroid
                    P = np.array([[-cx_m - tv_m/2, -cy_m - th_m/2 + lv_m], [-cx_m - tv_m/2, -cy_m - th_m/2], [-cx_m - tv_m/2 + lh_m, -cy_m - th_m/2]])
                    # rotate this points to the u-u, v-v coordinates system
                    D1 = np.array([[np.cos(Theta_vv1), np.sin(Theta_vv1)], [-np.sin(Theta_vv1), np.cos(Theta_vv1)]])
                    D2 = np.array([[np.cos(Theta_vv2), np.sin(Theta_vv2)], [-np.sin(Theta_vv2), np.cos(Theta_vv2)]])
                    Pp1 = np.dot(D1,np.transpose(P))
                    Pp2 = np.dot(D2,np.transpose(P))
                    
                    # calculate minimum section modulus
                    for pi in range(3):
                        Ppui1 = np.abs(Pp1[0, pi]) # get absolut u coordinate of point pi
                        Ppvi1 = np.abs(Pp1[1, pi]) # get absolut v coordinate of point pi
                        Ppui2 = np.abs(Pp2[0, pi]) # get absolut u coordinate of point pi
                        Ppvi2 = np.abs(Pp2[1, pi]) # get absolut v coordinate of point pi
                        if pi == 0:
                            W_uu1 = Juu_use1 / Ppvi1
                            W_vv1 = Jvv_use1 / Ppui1
                            W_uu2 = Juu_use2 / Ppvi2
                            W_vv2 = Jvv_use2 / Ppui2
                        if pi > 0:
                            if Ppvi1 != 0:
                                if W_uu1 > Juu_use1 / Ppvi1:
                                   W_uu1 = Juu_use1 / Ppvi1
                            if Ppui1 != 0:
                                if W_vv1 > Jvv_use1 / Ppui1:
                                   W_vv1 = Jvv_use1 / Ppui1
                            if Ppvi2 != 0:
                                if W_uu2 > Juu_use2 / Ppvi2:
                                   W_uu2 = Juu_use2 / Ppvi2
                            if Ppui2 != 0:
                                if W_vv2 > Jvv_use2 / Ppui2:
                                   W_vv2 = Jvv_use2 / Ppui2 
                    
                    MuuRk1 = W_uu1 * fyk
                    MvvRk1 = W_vv1 * fyk
                    MuuRk2 = W_uu2 * fyk
                    MvvRk2 = W_vv2 * fyk
                    
                    # ideal bending torsional buckling moment according to Petersen: Stahlbau page 381
                    if j1_loads[3] <= j2_loads[3]:
                        psi_M_uu = j1_loads[3]/j2_loads[3]
                    if j1_loads[3] > j2_loads[3]:
                        psi_M_uu = j2_loads[3]/j1_loads[3]
                    if psi_M_uu < -1:
                        psi_M_uu = -1
                    if psi_M_uu > 1:
                        psi_M_uu = 1
                    if j1_loads[4] <= j2_loads[4]:
                        psi_M_vv = j1_loads[4]/j2_loads[4]
                    if j1_loads[4] > j2_loads[4]:
                        psi_M_vv = j2_loads[4]/j1_loads[4]
                    if psi_M_vv < -1:
                        psi_M_vv = -1
                    if psi_M_vv > 1:
                        psi_M_vv = 1
                    
                    Zeta_uu = 1.77 - 0.77 * psi_M_uu
                    Zeta_vv = 1.77 - 0.77 * psi_M_vv
                    
                    Mcr_uu1 = Zeta_uu * np.pi/L_m * np.sqrt(E * Juu_use1 * G * XsecJ0_m)
                    Mcr_vv1 = Zeta_vv * np.pi/L_m * np.sqrt(E * Jvv_use1 * G * XsecJ0_m)
                    Mcr_uu2 = Zeta_uu * np.pi/L_m * np.sqrt(E * Juu_use2 * G * XsecJ0_m)
                    Mcr_vv2 = Zeta_vv * np.pi/L_m * np.sqrt(E * Jvv_use2 * G * XsecJ0_m)
                    
                    # torsional bending buckling support capacity according to EC 1993-1-1_2010 6.3.2
                    lambda_bar_LT_uu1 = np.sqrt(W_uu1 * fyk / Mcr_uu1)
                    lambda_bar_LT_vv1 = np.sqrt(W_vv1 * fyk / Mcr_vv1)
                    lambda_bar_LT_uu2 = np.sqrt(W_uu2 * fyk / Mcr_uu2)
                    lambda_bar_LT_vv2 = np.sqrt(W_vv2 * fyk / Mcr_vv2)
                    
                    Phi_LT_uu1 = 0.5 * (1 + alpha_LT * (lambda_bar_LT_uu1 - 0.2) + lambda_bar_LT_uu1**2)
                    Phi_LT_vv1 = 0.5 * (1 + alpha_LT * (lambda_bar_LT_vv1 - 0.2) + lambda_bar_LT_vv1**2)
                    Phi_LT_uu2 = 0.5 * (1 + alpha_LT * (lambda_bar_LT_uu2 - 0.2) + lambda_bar_LT_uu2**2)
                    Phi_LT_vv2 = 0.5 * (1 + alpha_LT * (lambda_bar_LT_vv2 - 0.2) + lambda_bar_LT_vv2**2)
                    
                    Chi_LT_uu1 = 1 / (Phi_LT_uu1 + np.sqrt(Phi_LT_uu1**2 - lambda_bar_LT_uu1**2))
                    Chi_LT_vv1 = 1 / (Phi_LT_vv1 + np.sqrt(Phi_LT_vv1**2 - lambda_bar_LT_vv1**2))
                    Chi_LT_uu2 = 1 / (Phi_LT_uu2 + np.sqrt(Phi_LT_uu2**2 - lambda_bar_LT_uu2**2))
                    Chi_LT_vv2 = 1 / (Phi_LT_vv2 + np.sqrt(Phi_LT_vv2**2 - lambda_bar_LT_vv2**2))
                    
                    if Chi_LT_uu1 > 1:
                        Chi_LT_uu1 = 1
                    if Chi_LT_vv1 > 1:
                        Chi_LT_vv1 = 1
                    if Chi_LT_uu2 > 1:
                        Chi_LT_uu2 = 1
                    if Chi_LT_vv2 > 1:
                        Chi_LT_vv2 = 1
                    
                    # get interaction factors according to method 2 of annex A in EC 1993-1-1_2010
                    NRk1 = fyk * A_use1
                    NRk2 = fyk * A_use2
                    
                    Cmu = 0.6 + 0.4 * psi_M_uu
                    if Cmu < 0.4:
                        Cmu = 0.4
                    Cmv = 0.6 + 0.4 * psi_M_vv
                    if Cmv < 0.4:
                        Cmv = 0.4
                    CmLT = Cmu
                    
                    # use the convention as tension force being negative for buckling proofs
                    NEd1 = - j1_loads[2]
                    NEd2 = - j2_loads[2]
                    # use the convention as tension force being positive and pressure being negative for strength analysis
                    NEd1_str = j1_loads[2] 
                    NEd2_str = j2_loads[2] 
                    
                    
                    kuu1 = Cmu * (1 + 0.6 * lambda_bar_uu1 * NEd1 / (Chi_uu1 * NRk1 / gamma_buckle))
                    if kuu1 > Cmu * (1 + 0.6 * NEd1 / (Chi_uu1 * NRk1 / gamma_buckle)):
                        kuu1 = Cmu * (1 + 0.6 * NEd1 / (Chi_uu1 * NRk1 / gamma_buckle))
                    kvu1 = (1 - 0.05* lambda_bar_vv1 * NEd1 / ((CmLT-0.25) * Chi_vv1 * NRk1 / gamma_buckle))
                    if kvu1 < (1 - 0.05 * NEd1 / ((CmLT-0.25) * Chi_vv1 * NRk1 / gamma_buckle)):
                        kvu1 = (1 - 0.05 * NEd1 / ((CmLT-0.25) * Chi_vv1 * NRk1 / gamma_buckle))
                    kvv1 = Cmv * (1 + 0.6 * lambda_bar_vv1 * NEd1 / (Chi_vv1 * NRk1 / gamma_buckle))
                    if kvv1 > Cmv * (1 + 0.6 * NEd1 / (Chi_vv1 * NRk1 / gamma_buckle)):
                        kvv1 = Cmv * (1 + 0.6 * NEd1 / (Chi_vv1 * NRk1 / gamma_buckle))
                    kuv1 = kvv1
                    
                    kuu2 = Cmu * (1 + 0.6 * lambda_bar_uu2 * NEd2 / (Chi_uu2 * NRk2 / gamma_buckle))
                    if kuu2 > Cmu * (1 + 0.6 * NEd2 / (Chi_uu2 * NRk2 / gamma_buckle)):
                        kuu2 = Cmu * (1 + 0.6 * NEd2 / (Chi_uu2 * NRk2 / gamma_buckle))
                    kvu2 = (1 - 0.05* lambda_bar_vv2 * NEd2 / ((CmLT-0.25) * Chi_vv2 * NRk2 / gamma_buckle))
                    if kvu2 < (1 - 0.05 * NEd2 / ((CmLT-0.25) * Chi_vv2 * NRk2 / gamma_buckle)):
                        kvu2 = (1 - 0.05 * NEd2 / ((CmLT-0.25) * Chi_vv2 * NRk2 / gamma_buckle))
                    kvv2 = Cmv * (1 + 0.6 * lambda_bar_vv2 * NEd2 / (Chi_vv2 * NRk2 / gamma_buckle))
                    if kvv2 > Cmv * (1 + 0.6 * NEd2 / (Chi_vv2 * NRk2 / gamma_buckle)):
                        kvv2 = Cmv * (1 + 0.6 * NEd2 / (Chi_vv2 * NRk2 / gamma_buckle))
                    kuv2 = kvv2
                    
                    if (NEd1 <= 0):
                        # set the proof equal to the proof of equation 6.44 in EC 1993-1-1_2010
                        Chi_uu1 = 1
                        Chi_vv1 = 1
                        kuu1 = 1
                        kuv1 = 1
                        kvu1 = 1
                        kvv1 = 1
                        
                    if (NEd2 <= 0):
                        # set the proof equal to the proof of equation 6.44 in EC 1993-1-1_2010
                        Chi_uu2 = 1
                        Chi_vv2 = 1
                        kuu2 = 1
                        kuv2 = 1
                        kvu2 = 1
                        kvv2 = 1
    
                    proof1LTuu1 = NEd1 / (Chi_uu1*NRk1/gamma_buckle) + kuu1 * (np.abs(M1_uu)) / (Chi_LT_uu1 * MuuRk1 / gamma_buckle) + kuv1 * (np.abs(M1_vv)) / (MvvRk1 / gamma_buckle)
                    proof2LTuu1 = NEd1 / (Chi_vv1*NRk1/gamma_buckle) + kvu1 * (np.abs(M1_uu)) / (Chi_LT_uu1 * MuuRk1 / gamma_buckle) + kvv1 * (np.abs(M1_vv)) / (MvvRk1 / gamma_buckle)                
                    proof1LTvv1 = NEd1 / (Chi_uu1*NRk1/gamma_buckle) + kuu1 * (np.abs(M1_uu)) / (MuuRk1 / gamma_buckle) + kuv1 * (np.abs(M1_vv)) / (Chi_LT_uu1 * MvvRk1 / gamma_buckle)
                    proof2LTvv1 = NEd1 / (Chi_vv1*NRk1/gamma_buckle) + kvu1 * (np.abs(M1_uu)) / (MuuRk1 / gamma_buckle) + kvv1 * (np.abs(M1_vv)) / (Chi_LT_uu1 * MvvRk1 / gamma_buckle)
    
                    proof1LTuu2 = NEd2 / (Chi_uu2*NRk2/gamma_buckle) + kuu2 * (np.abs(M2_uu)) / (Chi_LT_uu2 * MuuRk2 / gamma_buckle) + kuv2 * (np.abs(M2_vv)) / (MvvRk2 / gamma_buckle)
                    proof2LTuu2 = NEd2 / (Chi_vv2*NRk2/gamma_buckle) + kvu2 * (np.abs(M2_uu)) / (Chi_LT_uu2 * MuuRk2 / gamma_buckle) + kvv2 * (np.abs(M2_vv)) / (MvvRk2 / gamma_buckle)                
                    proof1LTvv2 = NEd2 / (Chi_uu2*NRk2/gamma_buckle) + kuu2 * (np.abs(M2_uu)) / (MuuRk2 / gamma_buckle) + kuv2 * (np.abs(M2_vv)) / (Chi_LT_uu2 * MvvRk2 / gamma_buckle)
                    proof2LTvv2 = NEd2 / (Chi_vv2*NRk2/gamma_buckle) + kvu2 * (np.abs(M2_uu)) / (MuuRk2 / gamma_buckle) + kvv2 * (np.abs(M2_vv)) / (Chi_LT_uu2 * MvvRk2 / gamma_buckle)  
    
                    max_buckle_util = np.max([proof1LTuu1, proof2LTuu1, proof1LTvv1, proof2LTvv1, proof1LTuu2, proof2LTuu2, proof1LTvv2, proof2LTvv2])
    
                    # proof for tension resistance through weakened cross sectional area through bolts
                    # assumed number of bolts > 3 according to DIN EN 1993-1-8_2010 3.10.3 (3.13)
    
                    # Cross Sectional class 4 yield strength proof according to EN 1993-1-1_2010 6.2.9.3 (6.44)
                    sigma_x1 = np.zeros([4,2], dtype = float)
                    sigma_x2 = np.zeros([4,2], dtype = float)
                    for pi in range(3):
                        pxi = np.transpose(P)[0, pi]
                        pyi = np.transpose(P)[1, pi]
                        sigma_x1[pi,0] = NEd1_str / A_use1 + (j1_loads[3]+dMxEd1) * pyi / Jxx_use1 + (j1_loads[4]+dMyEd1) * pxi / Jyy_use1
                        sigma_x1[pi,1] = sigma_x1[pi,0] * gamma_m / fyk
                        sigma_x2[pi,0] = NEd2_str  / A_use2 + (j2_loads[3]+dMxEd2) * pyi / Jxx_use2 + (j2_loads[4]+dMyEd2) * pxi / Jyy_use2
                        sigma_x2[pi,1] = sigma_x2[pi,0] * gamma_m / fyk
                        if pi == 0:
                            sigma_x1[3,1] = sigma_x1[pi,0] * gamma_m / fyk
                            sigma_x2[3,1] = sigma_x2[pi,0] * gamma_m / fyk
                        if pi > 0:
                            if np.abs(sigma_x1[pi,0] * gamma_m / fyk) > np.abs(sigma_x1[3,1]):
                                sigma_x1[3,1] = sigma_x1[pi,0] * gamma_m / fyk
                            if np.abs(sigma_x2[pi,0] * gamma_m / fyk) > np.abs(sigma_x2[3,1]):
                                sigma_x2[3,1] = sigma_x2[pi,0] * gamma_m / fyk
                    
                    max_stress_util = np.max([np.abs(sigma_x1[3,1]), np.abs(sigma_x2[3,1])])
                    
                    bee_list[x,1].utilization_a[dlci, mi, 0:3] = np.array([max_buckle_util, max_stress_util, np.max([max_buckle_util, max_stress_util])])
                    #print "Member ",mi,class3_m," \tproof1LTuu1: ", round(proof1LTuu1,3), ",         \tproof2LTuu1: ",round(proof2LTuu1,3), ", \tproof1LTvv1: ", round(proof1LTvv1,3), ", \tproof2LTvv1: ", round(proof2LTvv1,3), ", \tmax usage: ", round(np.max([proof1LTuu1, proof2LTuu1, proof1LTvv1, proof2LTvv1]),3)
    
    for x in range(0,len(bee_list[0:,0])):
        if bee_list[x,1].bee_type == bee_type:
            if kind_of_parameter_set_to_use == "old":
                bee_list[x,1].calculate_fitness(cycle)
            if kind_of_parameter_set_to_use == "new":
                bee_list[x,1].calculate_new_fitness(cycle) 
                                
    # delete all .SD.sum and .SD.out files to save hard drive space
    for x in range(0,len(bee_list[0:,0])):
        if bee_list[x,1].bee_type == bee_type:
            if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum"):
                os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.sum")
            if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out"):
                os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + ".SD.out")
            if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.sum"):
                os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.sum")
            if os.path.exists(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out"):
                os.remove(root_out + "IRLT_cycle" + str(cycle) + "_bee"+ str(bee_list[x,1].bee_ID) + "_static.SD.out")
    
    return bee_list


def set_local_new_parameter(bee_list,bee_type):
    # choose random other bee with the same type
    # build list with IDs of all bees with the same type
    same_type_bee_list = []
    for bee in bee_list[0:,1]:
        if bee.bee_type == bee_type:
            same_type_bee_list.append(bee)
    for bee in bee_list[0:,1]:
        if bee.bee_type == bee_type:
            # choose random neighbour bee
            neighbour_bee = choice(same_type_bee_list)
            while int(neighbour_bee.bee_ID) == int(bee.bee_ID):
                neighbour_bee = choice(same_type_bee_list)
            
            # choose random parameter
            param_ID = int(choice(bee.parameter_set_inst.parameters[0:,0]))
            
            """
            print "bee ", bee.bee_ID, " choose parameter ", param_ID, " of bee ", neighbour_bee.bee_ID
            print "bee ", neighbour_bee.bee_ID, " parameter ", param_ID, " = ", neighbour_bee.parameter_set_inst.parameters[param_ID,1]
            """
            
            # calculate new parameter value for parameter param_ID
            #print "bee ", bee.bee_ID, " parameter ", param_ID, " = ", bee.parameter_set_inst.parameters[param_ID,1]
            #print "parameter set: ", bee.print_parameters()
            param_new_value = bee.parameter_set_inst.parameters[param_ID,1] + uniform(-1,1) * (bee.parameter_set_inst.parameters[param_ID,1] - neighbour_bee.parameter_set_inst.parameters[param_ID,1])
            # round param_new_value to the next integer, if it is the number of segments parameter Nseg
            if param_ID == len(bee.parameter_set_inst.parameters[0:,0]) - 1:
                param_new_value = int(round(param_new_value,0))
            # set new parameter value to border, if it would exceed it            
            if param_new_value > bee.parameter_set_inst.parameters[param_ID,3]:
                param_new_value = bee.parameter_set_inst.parameters[param_ID,3]
            if param_new_value < bee.parameter_set_inst.parameters[param_ID,2]:
                param_new_value = bee.parameter_set_inst.parameters[param_ID,2]
            # set all parameters of the new parameter set to the values of the old parameterset
            bee.parameter_set_inst_new.parameter_set_change(np.asanyarray(list(bee.parameter_set_inst.parameters)))
            # change the parameter param_ID to the param_new_value
            bee.parameter_set_inst_new.parameter_change(param_ID,param_new_value)
            """
            print "new parameter ", param_ID, " of bee ", bee.bee_ID, " is now ", bee.parameter_set_inst_new.parameters[param_ID,1]
            print "old parameters of bee ", bee.bee_ID
            bee.parameter_set_inst.print_parameter_set()
            print "new parameters of bee ", bee.bee_ID
            bee.parameter_set_inst_new.print_parameter_set()
            """
    
    return bee_list

def compare_fitness(bee_list):   
    for bee in bee_list[0:,1]:
        if bee.bee_type == "employed bee" or bee.bee_type == "onlooker bee":
            if bee.fitness < bee.fitness_new:
                bee.local_cycle = 0
                bee.fitness_change(bee.fitness_new)
                bee.parameter_set_inst.parameter_set_change(np.asanyarray(list(bee.parameter_set_inst_new.parameters)))
                bee.set_obj_func_value(bee.obj_func_value_new)
    return bee_list

def get_probabilities(bee_list):
    # count employed bees
    n_prob_bees = 0
    for i in range(len(bee_list[:,1])):
        bee = bee_list[i,1]
        if bee.bee_type == "employed bee" or bee.bee_type == "scout bee":
            n_prob_bees += 1
    fitness_list = np.zeros([n_prob_bees, 2])
    prob_list = np.zeros([n_prob_bees, 2])
    z_emp_bee = 0
    for i in range(len(bee_list[:,1])):
        bee = bee_list[i,1]
        if bee.bee_type == "employed bee" or bee.bee_type == "scout bee":
            fitness_list[z_emp_bee,0] = bee.bee_ID
            fitness_list[z_emp_bee,1] = bee.fitness
            z_emp_bee += 1

    prob_sum = 0
    for j in range(len(fitness_list[0:,1])):
        prob_sum += fitness_list[j,1]
    for j in range(len(fitness_list[0:,1])):
        prob_list[j,0] = fitness_list[j,0]
        prob_list[j,1] = fitness_list[j,1] / prob_sum
    return fitness_list,prob_list

def onlookers_position(bee_list, prob_list):
    for i in range(len(bee_list[:,1])):
        bee = bee_list[i,1]
        if bee.bee_type == "onlooker bee":
            # get the parameter_set of one employed bee, dependent on their probability
            empl_bee_ID = weighted_choice(prob_list) # get the random ID of one employed bee
            for k in range(len(bee_list[:,1])):
                if bee_list[k,1].bee_ID == empl_bee_ID:
                    # give the onlooker bee the same parameter set like the employed bee
                    bee_list[i,1].parameter_set_inst.parameter_set_change(np.asanyarray(list(bee_list[k,1].parameter_set_inst.parameters)))
                    # set all parameters of the new parameter set to the values of the old parameterset       
                    bee_list[i,1].parameter_set_inst_new.parameter_set_change(np.asanyarray(list(bee_list[i,1].parameter_set_inst.parameters)))
                    bee_list[i,1].set_obj_func_value(bee_list[k,1].obj_func_value)
                    bee_list[i,1].set_obj_func_value_new(bee_list[k,1].obj_func_value)
                    bee_list[i,1].fitness_change(bee_list[k,1].fitness)
    return bee_list


def new_scout_position(bee_list, Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot):
    for i in range(len(bee_list[:,1])):
        bee = bee_list[i,1]
        if bee.bee_type == "scout bee":
            bee_list[i,1] = class_bee(i, "scout bee", Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot)
    return bee_list



### Artificial Bee Colony Optimization
# initialize bee colony
print "Initialize all bees"
if not initial_conditions:
    bee_list = create_bees(n_employed_bees, n_onlooker_bees, n_scout_bees, Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot)
if initial_conditions:
    bee_list = create_bees_initial_conditions(rootD, initial_cycle)

# calculate fitness of food sources of employed bees, onlooker bees and scout bee
bee_list = calculate_fitness_all(bee_list,"employed bee","old", 0)
#bee_list = calculate_fitness_all(bee_list,"onlooker bee","old", 0)
bee_list = calculate_fitness_all(bee_list,"scout bee","old", 0)

best_obj_func_value_glob = 1e6
best_parameter_set_glob = []
result_file = "IRLT__Results_" + start_dt + ".txt"
result_data = ["\n" for x in range(int(50e3))]
result_data[0] = 'ABC Rotatable Lattice Tower Results - Start Date ' + start_dt + "\n"
with open(root_out + result_file, 'w') as file:
    file.writelines(result_data)
    file.close()

best_ID = 0
for global_cycle in range(initial_cycle,n_cycles_max+1):
    obj_func_mean_list = []
    ### PLOT AREA
    best_obj_func_value = 1e7
    for i in range(len(bee_list[:,1])):
        obj_func_mean_list.append(bee_list[i,1].obj_func_value)
        
        if bee_list[i,1].obj_func_value <= best_obj_func_value:
            best_obj_func_value = np.copy(bee_list[i,1].obj_func_value)
            best_ID = bee_list[i,1].bee_ID
            best_bee_index = i
            best_parameter_set_inst = bee_list[i,1].parameter_set_inst
    print "\nbest bee of cycle ", global_cycle, " is bee ", best_ID, " with local cycle = ", bee_list[best_bee_index,1].local_cycle
    print "best obj_func_value = ", best_obj_func_value
    if best_obj_func_value <= best_obj_func_value_glob:
        best_obj_func_value_glob = np.copy(best_obj_func_value)
        best_parameter_set_glob = best_parameter_set_inst
    print "best obj_func_value_glob = ", best_obj_func_value_glob
    #print "best parameter_set_glob mean:", np.mean(best_parameter_set_glob.parameters[:,1])
    print "mean obj_func_value = ", np.mean(obj_func_mean_list)
    
    # write results to file:
    with open(root_out + result_file,'r') as file:
        prev_data = file.readlines()
        file.close()
    for lpd in range(len(prev_data)):
        if prev_data[lpd] == "\n":
            pd_i = lpd            
            break
    prev_data[pd_i] = "best bee of cycle "+ fs(global_cycle,4)+ " is bee "+ fs(best_ID,3)+ ", mass: "+ fs(bee_list[best_bee_index,1].mass,6)+ ", f1: "+ fs(bee_list[best_bee_index,1].natFreq[0],3)+ ", util_max: " + fs(np.max(bee_list[best_bee_index,1].utilization_a[:,:,2]),3)+ ", util_mean: " + fs(np.mean(bee_list[best_bee_index,1].utilization_a[:,:,2]),3) + ", local cycle = "+ fs(bee_list[best_bee_index,1].local_cycle,3) + ". best obj_func_value = "+ fs(best_obj_func_value,3) + ". mean obj_func_value = "+ fs(np.mean(obj_func_mean_list),3) + ". best obj_func_value_glob = "+ fs(best_obj_func_value_glob,3) + "\n"
    print prev_data[pd_i]
    with open(root_out + result_file,'w') as file:
        file.writelines(prev_data)
        file.close()
    
    
    # get probabilities for each food source
    lists = get_probabilities(bee_list)
    prob_list = lists[1]
    print "probabilities have been calculated..."
    # the onlooker bees decide for the food sources of the employed bees or scout bees dependent of probabilities of each food source
    bee_list = onlookers_position(bee_list, prob_list)
    print "onlooker bees have decided for their new position..."
    
    # vary one random parameter of each parameter set of each employed and onlooker bee to a semi random new value
    bee_list = set_local_new_parameter(bee_list,"employed bee")
    bee_list = set_local_new_parameter(bee_list,"onlooker bee")
    print "one semi random parameter has been changed for each bee..."
    print "start to calculate the fitness of each bee again..."
    # calculate the fitness of food source of each employed and onlooker bee with the new parameter set
    bee_list = calculate_fitness_all(bee_list,"employed bee","new", global_cycle)
    bee_list = calculate_fitness_all(bee_list,"onlooker bee","new", global_cycle)
    # update list and create fitness array
    for i in range(len(bee_list[:,0])):
        bee_list[i,2] = bee_list[i,1].bee_type
    # calculate standard deviation of fitness from bee swarm
    S, n_empl_bees = calc_standard_deviation(bee_list, "employed bee")
    print "standard deviation: ", round(S,4)
    # compare and replace new fitness with corresponding parameter set, if it is better
    bee_list = compare_fitness(bee_list)
    
    ### SCOUT AREA
    # get new random position for scout bee
    bee_list = new_scout_position(bee_list, Llmin, Llmax, tlmin, tlmax, Lbmin, Lbmax, tbmin, tbmax, Nseg_min, Nseg_max, alpha_y_min, alpha_y_max, sF_min, sF_max, a_Top, a_Bot)
    # calculate the fitness of food source of each scout bee with the random changed parameter set
    bee_list = calculate_fitness_all(bee_list,"scout bee","old", global_cycle)