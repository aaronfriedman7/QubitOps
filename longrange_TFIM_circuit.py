#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020

@author: Aaron + Ollie
"""

import StabilizerSet as StSet
import NQubitOps as NQO
import random
import numpy as np
import sys

Nsite = 10
depth = Nsite*5
Nmeasure = 20
Nhist = 100


if len(sys.argv) > 1:
	run_index = int(sys.argv[1])
else:
	run_index = 0


pvals = np.linspace(0.05, 1, num=Nmeasure)
av_EE = np.zeros(Nmeasure)

for rep in range(Nhist):
	print(rep)

	for p_index, p in enumerate(pvals):

		# create empty set of stabilizers
		S = StSet.StabilizerSet(Nsite)

		# initialize in sigma z product state with N "z" stabilizers
		for i in range(Nsite):
		    S.set_string(i, "z", left_pad=i)

		# list of sites for XX operator pairs
		sites = [n for n in range(Nsite)]

		# apply gates for t time steps
		for t in range(depth):

			# shuffle the site array to generate random pairs
			random.shuffle(sites)
			pairs = list(zip(*[iter(sites)]*2))
			
			# apply (randomly) nonlocal XX gates
			for ij in pairs:
				pos1, pos2 = ij
				S.random_XX_gate(pos1, pos2)

			# apply Z gates in all locations
			for i in range(Nsite):
				S.random_Z_gate(i)

			# perform measurements (sublattice A)
			for i in range(Nsite//2):
				if random.random() < p:
					S.measure_XX(2*i, 2*i+1)

			# perform measurements (sublattice B)
			for i in range(Nsite//2 - 1):
				if random.random() < p:
					S.measure_XX(2*i+1, 2*i+2)

		EE = S.entanglement_entropy()
		av_EE[p_index] += EE
		print("\t", EE)


print(av_EE / Nhist)
np.save('entanglement_L10_Nhist100_{}'.format(run_index), np.vstack((pvals, av_EE)))