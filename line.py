__author__ = 'Carlos'

import numpy as np
from collections import deque


class Line(object):

    BUFF_SIZE = 3

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        self.fit_coeffs = None
        self.fitx = None
        self.fity = None
        self.rad_curvature_m = None
        self.x_base = None

        self.buffer_n_fits = deque([], self.BUFF_SIZE)
        self.buffer_curvature = deque([], self.BUFF_SIZE)

    def update_values(self, detected, fit_coeffs, fitx, fity, rad_curvature_m):

        self.detected = detected
        self.fitx = fitx
        self.fity = fity

        self.update_fit_coeffs(detected, fit_coeffs)
        self.update_rad_curvature(detected, rad_curvature_m)
        self.update_x_base(detected)

        self.update_buffer_n_fits(fit_coeffs)
        self.update_buffer_curvature(rad_curvature_m)

    def update_fit_coeffs(self, detected, fit_coeffs):
        if detected:
            self.fit_coeffs = fit_coeffs
        else:
            if len(self.buffer_n_fits) > 0:
                self.fit_coeffs = self.buffer_n_fits[-1]
            else:
                self.fit_coeffs = fit_coeffs

    def update_rad_curvature(self, detected, rad_curvature_m):
        if detected:
            self.rad_curvature_m = rad_curvature_m
        else:
            if len(self.buffer_curvature) > 0:
                self.rad_curvature_m = self.buffer_curvature[-1]
            else:
                self.rad_curvature_m = rad_curvature_m

    def update_x_base(self, detected):
        if detected:
            self.x_base = self.fitx[0]

    def update_buffer_n_fits(self, fit_coeffs):
        self.buffer_n_fits.append(fit_coeffs)

    def update_buffer_x_base(self, x_base):
        self.buffer_x_base.append(x_base)

    def update_buffer_curvature(self, rad_curvature_m):
        self.buffer_curvature.append(rad_curvature_m)