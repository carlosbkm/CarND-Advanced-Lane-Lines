__author__ = 'Carlos'

import numpy as np
from collections import deque


class Line(object):

    BUFF_SIZE = 3

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        self.fit_coeffs = None
        self.fit_coeffs_m = None
        self.fitx = None
        self.fity = None
        self.rad_curvature_m = None
        self.x_base = None

        self.buffer_n_fits = deque([], self.BUFF_SIZE)
        self.buffer_curvature = deque([], self.BUFF_SIZE)
        self.buffer_x_base = deque()

    def update_values(self, detected, fit_coeffs, fit_coeffs_m, fitx, fity, rad_curvature_m):
        self.detected = detected
        self.fit_coeffs = fit_coeffs
        self.fit_coeffs_m = fit_coeffs_m
        self.fitx = fitx
        self.fity = fity
        self.rad_curvature_m = rad_curvature_m
        self.x_base = fitx[0]

        self.update_buffer_n_fits(fit_coeffs)
        self.update_buffer_curvature(rad_curvature_m)
        self.update_buffer_x_base(self.x_base)

    def update_buffer_n_fits(self, fit_coeffs):
        self.buffer_n_fits.append(fit_coeffs)

    def update_buffer_x_base(self, x_base):
        self.buffer_x_base.append(x_base)

    def update_buffer_curvature(self, rad_curvature_m):
        self.buffer_curvature.append(rad_curvature_m)