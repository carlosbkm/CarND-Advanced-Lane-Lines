__author__ = 'Carlos'

import numpy as np
from collections import deque


class Line(object):

    BUFF_SIZE = 10

    def __init__(self):

        # Needed for plotting
        self.linex = None
        self.liney = None

        self.buffer_coeffs = deque([], self.BUFF_SIZE)
        self.buffer_curvature = deque([], self.BUFF_SIZE)
        self.threshold_color = (255, 255, 255)
        self.detected = False

    def update_values(self, new_coeffs, linex, liney, rad_curvature_m):
        self.detected = True
        self.linex = linex
        self.liney = liney
        self.buffer_coeffs.append(new_coeffs)
        self.buffer_curvature.append(rad_curvature_m)

    def get_coeffs(self):
        return self.buffer_coeffs[-1]

    def get_curvature(self):
        return self.buffer_curvature[-1]

    def get_coeff_mean(self):
        return np.mean(self.buffer_coeffs, axis=0)

    def get_coeff_median(self):
        return np.median(self.buffer_coeffs, axis=0)

    def get_curve_mean(self):
        return np.mean(self.buffer_curvature, axis=0)


    @staticmethod
    def _calculate_percent(diff, old_value):
        return 100 * diff / old_value