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
