#!/usr/bin/env python
# coding: utf-8

import numpy as np

class RingLaser:

    def __init__(self, side=None, form=None, wl=None, lat=None):

        self.wavelength = wl
        self.latitude = lat


        if form == "triangle":
            self.area = 0.25 * side**2 * np.sqrt(3)
            self.perimeter = 3*side

        elif form == "square":
            self.area = side**2
            self.perimeter = 4*side

        self.length_of_day = 23*3600+56*60+4.1
        self.earth_rotation_rate = 2 * np.pi / self.length_of_day


    def get_sagnac_frequency(self):

        scale_factor = (4*self.area)/(self.wavelength*self.perimeter)
        projected_omega = self.earth_rotation_rate*np.sin(self.latitude/180.*np.pi)

        return scale_factor * projected_omega

    def get_scale_factor(self):

        return (4*self.area)/(self.wavelength*self.perimeter)

