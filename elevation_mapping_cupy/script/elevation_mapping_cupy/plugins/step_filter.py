#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List

from .plugin_manager import PluginBase


class StepFilter(PluginBase):

    def __init__(self, critiacl_cell_num: int = 4,
                 critical_value: float = 0.2,
                 first_window_radius: float = 0.04,
                 second_window_radius: float = 0.04,
                 cell_n: int = 100,
                 input_layer_name: str = "elevation", **kwargs):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.critiacl_cell_num = critiacl_cell_num
        self.critical_value = critical_value
        self.first_window_radius = first_window_radius
        self.second_window_radius = second_window_radius

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
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
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                print(h[i][j])
                print("hello")
        hs1 = h
        return hs1
