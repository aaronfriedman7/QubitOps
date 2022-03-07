#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020

@author: Aaron + Ollie
"""
import timeit
import StabilizerSet as StSet




M = 6
S = StSet.StabilizerSet(M)

#for i in range(M):
#    S.set_string(i, "z", left_pad=i)

S.set_string(0, "xeeeez")
S.set_string(1, "zeeeex")
S.set_string(2, "exeeze")
S.set_string(3, "ezeexe")
S.set_string(4, "eexzee")
S.set_string(5, "eezxee")

S.print_stabilizers()

print(S.entanglement_entropy())







S.random_Z_gate(0)
S.print_stabilizers()
