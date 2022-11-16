import numpy as np


# implementation of key-frame extraction method show in: Artist-Friendly Facial Animation Retargeting:
def graph_simplification_original(x: np.array, t: np.array):
    """
    This is unfinished! The aclgorithm is half implemented in Vanilla_staggered_poses.py
    The backtracking step still has to be implemented

    :param x:
    :param t:
    :return:
    """
    E2 = np.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0]):

