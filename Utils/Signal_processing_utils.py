import numpy as np


def dx_dt(x: np.array, dt: float = 1, method=1):
    # input:
    # shape of x should either be [number of timestep, number of attributes]
    # or simply [number of timestep].
    #
    # dt matters if you want derivative, if you just need relative difference dt can just be 1
    #
    # method 1 is forward difference, method 2 is central differences
    out_dx_dt = np.zeros(x.shape)
    if len(x.shape) == 2:
        for j in range(0, x.shape[1]):
            if method == 1:
                for i in range(0, x.shape[0] - 1):
                    out_dx_dt[i, j] = (x[i + 1, j] - x[i, j])/dt
                out_dx_dt[-1, j] = out_dx_dt[-2, j]
            if method == 2:
                for i in range(1, x.shape[0] - 1):
                    out_dx_dt[i, j] = (x[i + 1, j] - x[i - 1, j]) / 2 / dt
                out_dx_dt[-1, j] = out_dx_dt[-2, j]
                out_dx_dt[0, j] = out_dx_dt[1, j]
    elif len(x.shape) == 1:
        if method == 1:
            for i in range(0, x.shape[0] - 1):
                out_dx_dt[i] = (x[i + 1] - x[i]) / dt
            out_dx_dt[-1] = out_dx_dt[-2]
        if method == 2:
            for i in range(1, x.shape[0] - 1):
                out_dx_dt[i] = (x[i + 1] - x[i - 1]) / 2 / dt
            out_dx_dt[-1] = out_dx_dt[-2]
            out_dx_dt[0] = out_dx_dt[1]
    return out_dx_dt

