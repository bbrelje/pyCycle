# --- Python 3.8 ---
# FileName: N3_v1_pressure_sweep.py
# Created by: alamkin
# Date: 7/16/20
# Last Updated: 10:37 AM

# --- Imports ---
import os
import time
import numpy as np 



if __name__ == "__main__":

    mach3d = np.zeros((8,7,11)) 
    alt3d = np.zeros((8,7,11)) 
    throt3d = np.zeros((8,7,11)) 

    throttlelist = [1.0, 0.8, 0.6, 0.4, 0.2, 0.10, 0.05, 0.3, 0.5, 0.7, 0.9]
    for i, alt in enumerate([35000., 30000., 25000., 20000., 15000., 10000., 5000., 0.]):
        for j, MN in enumerate([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]):
            for k, throttlefrac in enumerate(throttlelist):
                mach3d[i,j,k] = MN
                alt3d[i,j,k] = alt
                throt3d[i,j,k] = throttlefrac
                np.save('mach', mach3d)
                np.save('alt', alt3d)
                np.save('throttle', throt3d)

        
