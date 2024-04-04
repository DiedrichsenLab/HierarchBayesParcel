#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to test Dice coefficient

Created on 4/4/2024
Author: dzhi
"""
import torch as pt
import numpy as np
from sklearn import metrics
import evaluation as ev


if __name__ == '__main__':
    l1 = pt.tensor([1,2,2,1,3,3,3,2,2,1])
    l2 = pt.tensor([2,1,2,3,3,1,3,1,2,2])
    l3 = pt.tensor([1,2,1,3,3,2,3,2,1,1])

    dice  = ev.dice_coefficient(l1, l2)
    print("Dice 1: ", dice)
    from torchmetrics.classification import Dice
    metric = Dice(average='macro', num_classes=3)
    print('Dice 2:', metric(l1-1, l3-1))