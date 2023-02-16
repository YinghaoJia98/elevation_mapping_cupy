#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List

from .plugin_manager import PluginBase


class StepFilter(PluginBase):

    def __init__(self, critical_cell_num: int = 4,
                 critical_value: float = 0.2,
                 first_window_radius: float = 0.04,
                 second_window_radius: float = 0.04,
                 map_resolution: float = 0.02,
                 cell_n: int = 100,
                 input_layer_name: str = "elevation", **kwargs):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.critical_cell_num = critical_cell_num
        self.critical_value = critical_value
        self.first_window_radius = first_window_radius
        self.second_window_radius = second_window_radius
        self.map_resolution = map_resolution

    def __call__(self, elevation_map: cp.ndarray,
                 layer_names: List[str], plugin_layers: cp.ndarray,
                 plugin_layer_names: List[str],
                 ) -> cp.ndarray:
        print(layer_names)
        print(plugin_layer_names)
        if self.input_layer_name in layer_names:
            idx = layer_names.index(self.input_layer_name)
            h = elevation_map[idx]
        elif self.input_layer_name in plugin_layer_names:
            idx = plugin_layer_names.index(self.input_layer_name)
            h = plugin_layers[idx]
        else:
            print("layer name {} was not found. Using elevation layer.".format(
                self.input_layer_name))
            h = elevation_map[0]
        h_step = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        # print("dierguan")
        h_score = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        print("diyiguan")
        fisrt_num = self.first_window_radius/self.map_resolution
        second_num = self.second_window_radius/self.map_resolution
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if (cp.isnan(h[i][j])):
                    # if (h[i][j] == 0.0000000): The nan value might be 0.000000 in ndarray and can not be distinguished
                    continue
                height = h[i][j]
                init: bool = False
                k1: int = 0
                heightMax: float = 0.0
                heightMin: float = 0.0
                while k1 < (fisrt_num+1):
                    k2: int = -k1
                    while k2 < (k1+1):
                        if (cp.isnan(h[k1][k2])):
                            # if (h[k1][k2] == 0.000000):
                            print(h[k1][k2])
                            continue
                        if (not init):
                            heightMax = h[k1][k2]
                            heightMin = h[k1][k2]
                            init = True
                            continue
                        if height > heightMax:
                            heightMax = height
                        if height < heightMin:
                            heightMin = height
                        k2 += 1
                    k1 += 1
                if init:
                    step_height = heightMax-heightMin
                    h_step[i][j] = step_height
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                nCells: int = 0
                stepMax: float = 0.0
                isValid: bool = False
                k1: int = 0
                while k1 < (second_num+1):
                    k2: int = -k1
                    while k2 < (k1+1):
                        if cp.isnan(h_step[k1][k2]):
                            # if (h_step[k1][k2] == 0.000000):
                            print(h_step[k1][k2])
                            continue
                        isValid = True
                        if h_step[k1][k2] > stepMax:
                            stepMax = h_step[k1][k2]
                        if h_step[k1][k2] > self.critical_value:
                            nCells += 1
                        k2 += 1
                    k1 += 1
                if isValid:
                    nCells_float = float(nCells)
                    critical_cell_num_float = float(self.critical_cell_num)
                    step_middle = nCells_float / critical_cell_num_float*stepMax
                    step = min(stepMax, step_middle)
                    if (step < self.critical_value):
                        h_score[i][j] = 1.0-step/self.critical_value
                    else:
                        h_score[i][j] = 0.0
        hs1 = h
        # print(h_score)
        return h
