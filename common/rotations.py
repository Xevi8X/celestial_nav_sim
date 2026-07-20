import numpy as np

class Rotation:
    @staticmethod
    def X(fi):
        c, s = np.cos(fi), np.sin(fi)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def Y(fi):
        c, s = np.cos(fi), np.sin(fi)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def Z(fi):
        c, s = np.cos(fi), np.sin(fi)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    @staticmethod
    def align(source, target):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)

        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source and target must have shape (N, 3)")

        source /= np.linalg.norm(source, axis=1, keepdims=True)
        target /= np.linalg.norm(target, axis=1, keepdims=True)

        U, _, Vt = np.linalg.svd(target.T @ source)

        if np.linalg.det(U @ Vt) < 0:
            U[:, -1] *= -1

        return U @ Vt
