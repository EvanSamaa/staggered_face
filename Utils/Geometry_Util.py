import numpy as np


def rotation_angles_frome_positions(arr):
    """
    converts an array of positions to an array of rotation angles (azimuth, elevation)
    centered at the origin, where:
        azimuth: +right,-left
        elevation: +up,-down
    here we assume that the input vectors are in world coordinates
    :param arr: array with shape (N, 3)
    :return: array with shape (N, 2)
    """
    # F: arr (N, 3) -> arr (N, 2)
    # in the output is in the convention of (azimuth, elevation)

    mag = np.sqrt(np.sum(arr * arr, axis=1, keepdims=True))

    out = arr / mag
    out[:, 0] = np.arcsin(out[:, 0])
    out[:, 1] = np.arcsin(out[:, 1])

    return out[:, 0:2] * 180 / np.pi
def directions_from_rotation_angles(arr, magnitudes):
    """
    converts an array of rotation angles (in degrees) to an array of positions (x, y, z)
    :param arr: array of rotations (centered at the origins) shape of (N, 2)
    :param magnitudes: magnitude of the direction vector, shape of (N, 1)
    :return: array with shape of (N, 3)
    """
    out = np.ones((arr.shape[0], 3))
    out[:, 0] = magnitudes * np.tan(arr[:, 0] / 180 * np.pi)
    out[:, 1] = magnitudes * np.tan(arr[:, 1] / 180 * np.pi)
    out[:, 2] = magnitudes * out[:, 2]
    return out

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# def position_world_to_local(pt_in_local, local_space_point):
