#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List
from .step_kernels import compute_hstep_kernel
from .step_kernels import compute_hscore_kernel

from .plugin_manager import PluginBase


class StepFilter(PluginBase):

    def __init__(self, critical_cell_num: int = 4,
                 critical_value: float = 0.30,
                 first_window_radius: float = 0.06,
                 second_window_radius: float = 0.06,
                 map_resolution: float = 0.03,
                 cell_n: int = 100,
                 input_layer_name: str = "elevation", **kwargs):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.critical_cell_num = critical_cell_num
        self.critical_value = critical_value
        self.first_window_radius = first_window_radius
        self.second_window_radius = second_window_radius
        self.map_resolution = map_resolution
        self.fisrt_num: int = int(first_window_radius/map_resolution)
        self.second_num: int = int(second_window_radius/map_resolution)
        self.compile_step_kernels()

    def compile_step_kernels(self):
        self.compute_hstep_kernel = compute_hstep_kernel(
            self.fisrt_num
        )
        self.compute_hscore_kernel = compute_hscore_kernel(
            self.second_num,
            self.critical_value,
            self.critical_cell_num
        )

    def __call__(self, elevation_map: cp.ndarray,
                 layer_names: List[str], plugin_layers: cp.ndarray,
                 plugin_layer_names: List[str],
                 ) -> cp.ndarray:
        # print(layer_names)
        # print(plugin_layer_names)
        # seconds_qian=time.time()
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
        # ElementwiseKernel might be helpful, try it_20230215
        h = cp.where(elevation_map[2] > 0.5, h, cp.nan)
        h_step = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        h_score = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        self.compute_hstep_kernel(h, h_step)
        # print(h.shape)
        # print(h_step.shape)
        # print(h_step)
        self.compute_hscore_kernel(h_step, h_score)
        # print(self.critical_value)

        # seconds_hou=time.time()
        # seconds_used=seconds_hou-seconds_qian
        # print(seconds_used)
        # hs1 = h
        # print(h_score)
        return h_score
