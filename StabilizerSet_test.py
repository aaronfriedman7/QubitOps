#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020

@author: Aaron + Ollie
"""
import time
import StabilizerSet as StSet
import NQubitOps as NQO
import numpy as np


# entanglement entropy of a state
def State_EntEnt(L, vec, cut):
    lower = cut
    upper = L - cut
    Psi = np.reshape(vec,(2**(lower),2**(upper)))
    lambdas = np.linalg.svd(Psi, compute_uv = False)
    EntEnt = 0.0
    for foo in range(len(lambdas)):
        value = (lambdas[foo])**2
        if value > 0.0:
            EntEnt -= np.math.log(value)*value
    return EntEnt

def EntEnt(rho, L, cut):
    assert cut < L and cut > 0
    rhoAB = np.reshape(rho,(2**cut,2**(L-cut),2**cut,2**(L-cut)))
    RDM = np.trace(rhoAB, offset=0, axis1=1, axis2=3)
    probs, states = np.linalg.eigh(RDM)
    S = - np.sum(probs[probs>1e-20]*np.log(probs[probs>1e-20]))
    return S

def Renyi0(rho, L, cut):
    assert cut < L and cut > 0
    rhoAB = np.reshape(rho,(2**cut,2**(L-cut),2**cut,2**(L-cut)))
    RDM = np.trace(rhoAB, offset=0, axis1=1, axis2=3)
    RDM = RDM**0
    S = np.log(np.trace(RDM))
    return S

def Renyi2(rho, L, cut):
    assert cut < L and cut > 0
    rhoAB = np.reshape(rho,(2**cut,2**(L-cut),2**cut,2**(L-cut)))
    RDM = np.trace(rhoAB, offset=0, axis1=1, axis2=3)
    RDM = np.matmul(RDM,RDM)
    S = - np.log(np.trace(RDM))
    return S


def RenyiN(rho, L, cut, n):
    assert cut < L and cut > 0
    assert n != 1
    rhoAB = np.reshape(rho,(2**cut,2**(L-cut),2**cut,2**(L-cut)))
    RDM = np.trace(rhoAB, offset=0, axis1=1, axis2=3)
    m=1
    placeholder = RDM.copy()
    while m < n:
        placeholder = np.matmul(placeholder,RDM)
        m += 1
    S = np.log(np.trace(RDM))/(1.0-float(n))
    return S


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



start = time.time()
XX = NQO.PauliStr.from_char_string("xxeeeeeeeeeeeeee")

for i in range(num):
	S_i = NQO.PauliStr(num, S3.coeff[i], S3.X_arr[i, :], S3.Z_arr[i, :])

	if NQO.PauliStr.check_comm(XX, S_i):
		S_i.print_string()
	else:
		prod = NQO.PauliStr.product(XX, S_i)
		prod.rescale(1j)
		prod.print_string()
end = time.time()

print("elapsed time = {}".format(end-start))
print()
L=8
# stabilizers
Gs = []
g_strings = ["zeeeeeee", "zyeyxeex", "eezeeeee", "zxeeeeey", "eeezxeey", "eeeeezee", "eeeeeeze", "eeeyzeey"]

for thong in g_strings:
    newPauli = NQO.PauliStr.from_char_string(thong)
    Gs.append(newPauli)


densitymatrix = StSet.dmat_array(L, Gs)
print("new density matrix has trace = {}".format(np.trace(densitymatrix)))

S = EntEnt(densitymatrix,8,4)
print("vN entanglement entropy is {} and log2 is {}".format(S,np.log(2.0)))


S0 = Renyi0(densitymatrix,8,4)
print("Renyi 0 is {}".format(S0))

S2 = Renyi2(densitymatrix,8,4)
print("Renyi 2 is {}".format(S2))
