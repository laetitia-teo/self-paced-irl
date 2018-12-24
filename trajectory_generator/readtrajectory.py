# ====================================================================
# Small utility function for reading the trajectory data from a file.
# ====================================================================

import ast

def read(path):
    traj = []
    with open(path) as f:
        for line in f.readlines():
            t = ast.literal_eval(line)
            traj.append(t)
    return traj
