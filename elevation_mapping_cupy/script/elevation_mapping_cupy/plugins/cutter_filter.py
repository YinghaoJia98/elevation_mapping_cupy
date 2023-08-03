#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List
from .cutter_kernels import cutter_min_kernel


from .plugin_manager import PluginBase


class CutterFilter(PluginBase):

    def __init__(self, cell_n: int = 100,
                 first_num: int = 2,
                 MaxGap: float = 0.25,
                 CutterInputLayer: str = "elevation", **kwargs):
        super().__init__()
        self.input_layer_name = CutterInputLayer
        self.fisrt_num = first_num
        self.MaxGap = MaxGap
        self.compile_cutter_kernels()

    def compile_cutter_kernels(self):
        self.compute_cutter_kernel = cutter_min_kernel(
            self.MaxGap,
            self.fisrt_num
        )


    def __call__(self, elevation_map: cp.ndarray,
                 layer_names: List[str],
                 plugin_layers: cp.ndarray,
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
            print("[cutter_info] layer name {} was not found. Using elevation layer.".format(
                self.input_layer_name))
            h = elevation_map[0]
        # ElementwiseKernel might be helpful, try it_20230215
        h = cp.where(elevation_map[2] > 0.5, h, cp.nan)
        h_cuttered = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        h_cuttered2 = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        self.compute_cutter_kernel(h, h_cuttered)
        self.compute_cutter_kernel(h_cuttered, h_cuttered2)
        # h_cuttered=h
        # print(h.shape)
        # print(h_step.shape)
        # print(h_step)
        return h_cuttered2