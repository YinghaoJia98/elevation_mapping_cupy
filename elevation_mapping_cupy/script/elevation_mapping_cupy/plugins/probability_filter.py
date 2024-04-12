#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List
from .sample_kernels import normalize_kernel
from .sample_kernels import merge_traversability_kernel
from .sample_kernels import filtered_traversability_kernel
import cv2
import numpy as np
# import time



from .plugin_manager import PluginBase


class SampleFilter(PluginBase):

    def __init__(self, cell_n: int = 100,
                 blur_radiu_num: int = 51,
                 sigma_num: int = 25,
                 StepThreshold: float = 0.7,
                 SampleInputLayer1: str = "elevation",
                 SampleInputLayer2: str = "step", **kwargs):
        super().__init__()
        self.SampleInputLayer1 = SampleInputLayer1
        self.SampleInputLayer2 = SampleInputLayer2
        self.blur_radiu_num = blur_radiu_num
        self.sigma_num = sigma_num
        self.StepThreshold = StepThreshold
        self.compile_sample_kernels()

    def compile_sample_kernels(self):
        self.compute_normalize_kernel = normalize_kernel(
            self.StepThreshold
        )
        self.Comput_merge_kernel=merge_traversability_kernel(
            self.StepThreshold
        )
        self.Comput_Filtered_Kernel=filtered_traversability_kernel(
            self.StepThreshold
        )


    def __call__(self, elevation_map: cp.ndarray,
                 layer_names: List[str],
                 plugin_layers: cp.ndarray,
                 plugin_layer_names: List[str],
                 ) -> cp.ndarray:
        # print(layer_names)
        # print(plugin_layer_names)
        # seconds_qian=time.time()

        if self.SampleInputLayer1 in layer_names:
            idx = layer_names.index(self.SampleInputLayer1)
            h = elevation_map[idx]
        elif self.SampleInputLayer1 in plugin_layer_names:
            idx = plugin_layer_names.index(self.SampleInputLayer1)
            h = plugin_layers[idx]
        else:
            print("[sample1_info] layer name {} was not found. Using elevation layer.".format(
                self.SampleInputLayer1))
            h = elevation_map[0]

        if self.SampleInputLayer2 in layer_names:
            idx = layer_names.index(self.SampleInputLayer2)
            step = elevation_map[idx]
        elif self.SampleInputLayer2 in plugin_layer_names:
            idx = plugin_layer_names.index(self.SampleInputLayer2)
            step = plugin_layers[idx]
        else:
            print("[sample2_info] layer name {} was not found. Using elevation layer.".format(
                self.SampleInputLayer2))
            step = elevation_map[0]
        # ElementwiseKernel might be helpful, try it_20230215
        h = cp.where(elevation_map[2] > 0.5, h, cp.nan)
        h_normalized = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        self.compute_normalize_kernel(h,step,h_normalized)
        h_normalized_np = cp.asnumpy(h_normalized)
        kernel_size = (self.blur_radiu_num, self.blur_radiu_num)
        sigma = self.sigma_num
        h_probability_np=cv2.GaussianBlur(h_normalized_np,kernel_size,sigma)
        h_probability2_np=np.amax(h_probability_np)-h_probability_np
        h_probability2_cp=cp.asarray(h_probability2_np)
        h_merged = cp.empty((h.shape[0], h.shape[1]), dtype=float)
        self.Comput_merge_kernel(h,h_probability2_cp,h_normalized,h_merged)
        h_merged_np=cp.asnumpy(h_merged)
        h_range=np.amax(h_merged_np)-np.min(h_merged_np)
        h_merged_np=0.5+h_merged_np*0.5/h_range
        h_merged_np_cp=cp.asarray(h_merged_np)
        h_merged_revised=cp.empty((h.shape[0], h.shape[1]), dtype=float)
        self.Comput_Filtered_Kernel(h,h_merged_np_cp,h_normalized,h_merged_revised)
        return h_merged_revised