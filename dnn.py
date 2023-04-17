
import functools as ft 

import torch.nn as nn 
from typing import List 

class MLPNetwork(nn.Module):
    def __init__(self, layers_config:List[int]):
        super(MLPNetwork, self).__init__()
        self.shapes = list(zip(layers_config[:-1], layers_config[1:]))
        self.layers = nn.ModuleList([])
        for shape in self.shapes[:-1]:
            self.layers.append(nn.Sequential(
                nn.Linear(shape[0], shape[1]),
                nn.ReLU(),
                nn.BatchNorm1d(shape[1])
            ))
        
        self.layers.append(nn.Linear(self.shapes[-1][0], self.shapes[-1][1]))
    
    def forward(self, X0):
        XN = ft.reduce(
            lambda Xi, current_layer: current_layer(Xi),
            self.layers,
            X0
        )
        return XN 

        