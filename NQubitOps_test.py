#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:07:38 2022

@author: Aaron + Ollie
"""

import numpy as np
import NQubitOps as NQO
import timeit




# create two long strings
PA = NQO.PauliStr.from_char_string('xyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyze', coeff=1)
PB = NQO.PauliStr.from_char_string('yzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzex', coeff=1)

# time the multiplication (averaged over 100000 repeats)
print(timeit.timeit('PA.dot(PB)', globals=globals(), number=100000))


print("Single qubit operator tests")
