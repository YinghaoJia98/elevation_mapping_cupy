#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List
from .frontier_kernels import frontier_kernel

# import time



from .plugin_manager import PluginBase


class SampleFilter(PluginBase):

    def __init__(self,
                 TraThreshold: float = 0.7,
                 FrontierInputLayer1: str = "elevation",
                 FrontierInputLayer2: str = "step_cutter", **kwargs):
        super().__init__()
        self.FrontierInputLayer1 = FrontierInputLayer1
        self.FrontierInputLayer2 = FrontierInputLayer2
        self.TraThreshold = TraThreshold

        self.compile_frontier_kernels()

    def compile_frontier_kernels(self):
        self.compute_frontier_kernel = frontier_kernel(
            self.TraThreshold
        )

    def __call__(self, elevation_map: cp.ndarray,
                 layer_names: List[str],
                 plugin_layers: cp.ndarray,
                 plugin_layer_names: List[str],
                 ) -> cp.ndarray:
        # print(layer_names)
        # print(plugin_layer_names)
        # seconds_qian=time.time()

        if self.FrontierInputLayer1 in layer_names:
            idx = layer_names.index(self.FrontierInputLayer1)
            h = elevation_map[idx]
        elif self.FrontierInputLayer1 in plugin_layer_names:
            idx = plugin_layer_names.index(self.FrontierInputLayer1)
            h = plugin_layers[idx]
        else:
            print("[sample1_info] layer name {} was not found. Using elevation layer.".format(
                self.FrontierInputLayer1))
            h = elevation_map[0]

        if self.FrontierInputLayer2 in layer_names:
            idx = layer_names.index(self.FrontierInputLayer2)
            step = elevation_map[idx]
        elif self.FrontierInputLayer2 in plugin_layer_names:
            idx = plugin_layer_names.index(self.FrontierInputLayer2)
            step = plugin_layers[idx]
        else:
            print("[sample2_info] layer name {} was not found. Using elevation layer.".format(
                self.FrontierInputLayer2))
            step = elevation_map[0]
        # ElementwiseKernel might be helpful, try it_20230215
        h = cp.where(elevation_map[2] > 0.5, h, cp.nan)
        step = cp.where(elevation_map[2] > 0.5, step, cp.nan)
        h_frontier = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        self.compute_frontier_kernel(h,step,h_frontier)
        #print(h_frontier)

        # cum_prob_rowwise_hack_cp_=cp.asarray(cum_prob_rowwise_hack)
        # seconds_used=time.time()-seconds_qian
        # print(seconds_used)
        # h_cuttered=h
        # print(h.shape)
        # print(h_step.shape)
        # print(h_step)
        return h_frontier