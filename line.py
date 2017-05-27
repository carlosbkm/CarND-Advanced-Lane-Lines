__author__ = 'Carlos'

import numpy as np
from collections import deque
class Line(object):

    BUFF_SIZE = 5
    THRESHOLD = [np.float64('1.5e-05'), np.float64('1.5e-02')]
    PERCENTAGE_THRESHOLD = [np.float64(30), np.float64(30), np.float64(10)]

    def __init__(self):

        # Needed for plotting
        self.fitx = None
        self.fity = None

        self.buffer_coeffs = deque([], self.BUFF_SIZE)
        self.buffer_curvature = deque([], self.BUFF_SIZE)
        self.buffer_diff = deque([], self.BUFF_SIZE)
        self.threshold_color = (255, 255, 255)
        self.detected = False

    def update_values(self, new_coeffs, fitx, fity, rad_curvature_m):
        self.fitx = fitx
        self.fity = fity

        np.concatenate((new_coeffs, np.array([fitx[0]])))

        if len(self.buffer_coeffs) == 0:
            self.buffer_coeffs.append(new_coeffs)
            self.buffer_curvature.append(rad_curvature_m)
            self.buffer_diff.append([0,0,0])
        else:
            last_coeff = self.buffer_coeffs[-1]
            # last_curve = self.buffer_curvature[-1]
            #last_diff = self.buffer_diff[-1]

            diff = new_coeffs - last_coeff
            diff_percent = [Line._calculate_percent(diff[0], last_coeff[0]),
                            Line._calculate_percent(diff[1], last_coeff[1]),
                            Line._calculate_percent(diff[2], last_coeff[2])]
                            # Line._calculate_percent(diff[3], last_coeff[3])]


            if self._meets_threshold(diff_percent):
                self.buffer_coeffs.append(new_coeffs)
                self.buffer_curvature.append(rad_curvature_m)
                self.threshold_color = (255, 255, 255)
                self.detected = True
            else:
                self.buffer_coeffs.append(np.mean(self.buffer_coeffs, axis=0))
                self.buffer_curvature.append(np.mean(self.buffer_curvature, axis=0))
                self.threshold_color = (255, 0, 0)
                self.detected = False
            self.buffer_diff.append(diff_percent)

    def get_coeffs(self):
        return self.buffer_coeffs[-1]

    def get_curvature(self):
        return self.buffer_curvature[-1]

    def get_diff(self):
        return self.buffer_diff[-1]

    def get_diff_mean(self):
        return np.mean(self.buffer_diff, axis=0)

    def get_diffx_mean(self):
        return np.mean(self.buffer_fitx, axis=0)

    def get_coeff_mean(self):
        return np.mean(self.buffer_coeffs, axis=0)

    def get_curve_mean(self):
        return np.mean(self.buffer_curvature, axis=0)

    # def get_x_base(self):
    #     return self.buffer_coeffs[3]
    #
    # def get_x_base_mean(self):
    #     return np.mean(self.buffer_coeffs, axis=0)[3]

    @staticmethod
    def _calculate_percent(diff, old_value):
        return np.abs(100 * diff / old_value)

    def _meets_threshold(self, diff_percent):
        if diff_percent[0] >= self.PERCENTAGE_THRESHOLD[0]:
            return False
        if diff_percent[1] >= self.PERCENTAGE_THRESHOLD[1]:
            return False
        if diff_percent[2] >= self.PERCENTAGE_THRESHOLD[2]:
            return False
        # if diff_percent[3] >= self.PERCENTAGE_THRESHOLD[3]:
        #     return False
        return True