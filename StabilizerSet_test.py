#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020

@author: Aaron + Ollie
"""
import timeit
import StabilizerSet as StSet




M = 6
S1 = StSet.StabilizerSet(M)

#for i in range(M):
#    S.set_string(i, "z", left_pad=i)

S1.set_string(0, "xeeeez")
S1.set_string(1, "zeeeex")
S1.set_string(2, "exeeze")
S1.set_string(3, "ezeexe")
S1.set_string(4, "eexzee")
S1.set_string(5, "eezxee")

S1.print_stabilizers()

# compute entanglement entropy
print(S1.entanglement_entropy())

# apply random gate on site 0
S1.random_Z_gate(0)
S1.print_stabilizers()




num = 16
S2 = StSet.StabilizerSet(num)

S2.set_string(0,  "eeeeeeeeeeeeeeee")
S2.set_string(1,  "xeeeeeeeeeeeeeee")
S2.set_string(2,  "zeeeeeeeeeeeeeee")
S2.set_string(3,  "yeeeeeeeeeeeeeee")

S2.set_string(4,  "exeeeeeeeeeeeeee")
S2.set_string(5,  "xxeeeeeeeeeeeeee")
S2.set_string(6,  "zxeeeeeeeeeeeeee")
S2.set_string(7,  "yxeeeeeeeeeeeeee")

S2.set_string(8,  "ezeeeeeeeeeeeeee")
S2.set_string(9,  "xzeeeeeeeeeeeeee")
S2.set_string(10, "zzeeeeeeeeeeeeee")
S2.set_string(11, "yzeeeeeeeeeeeeee")

S2.set_string(12, "eyeeeeeeeeeeeeee")
S2.set_string(13, "xyeeeeeeeeeeeeee")
S2.set_string(14, "zyeeeeeeeeeeeeee")
S2.set_string(15, "yyeeeeeeeeeeeeee")

S2.random_XX_gate(0, 1, gate=3)
S2.print_stabilizers()


num = 16
S3 = StSet.StabilizerSet(num)

S3.set_string(0,  "eeeeeeeeeeeeeeee")
S3.set_string(1,  "xeeeeeeeeeeeeeee")
S3.set_string(2,  "zeeeeeeeeeeeeeee")
S3.set_string(3,  "yeeeeeeeeeeeeeee")

S3.set_string(4,  "exeeeeeeeeeeeeee")
S3.set_string(5,  "xxeeeeeeeeeeeeee")
S3.set_string(6,  "zxeeeeeeeeeeeeee")
S3.set_string(7,  "yxeeeeeeeeeeeeee")

S3.set_string(8,  "ezeeeeeeeeeeeeee")
S3.set_string(9,  "xzeeeeeeeeeeeeee")
S3.set_string(10, "zzeeeeeeeeeeeeee")
S3.set_string(11, "yzeeeeeeeeeeeeee")

S3.set_string(12, "eyeeeeeeeeeeeeee")
S3.set_string(13, "xyeeeeeeeeeeeeee")
S3.set_string(14, "zyeeeeeeeeeeeeee")
S3.set_string(15, "yyeeeeeeeeeeeeee")

XX = NQO.PauliStr.from_char_string("xxeeeeeeeeeeeeee")

for i in range(num):
	S_i = NQO.PauliStr(num, S3.coeff[i], S3.X_arr[i, :], S3.Z_arr[i, :])
	isCommuting = NQO.PauliStr.check_comm(XX, S_i)

	if isCommuting:
		S_i.print_string()
	else:
		prod = NQO.PauliStr.product(XX, S_i)
		prod.rescale(1j)
		prod.print_string()

