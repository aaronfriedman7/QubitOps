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
# PA = NQO.PauliStr.from_char_string('xyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyze', coeff=1)
# PB = NQO.PauliStr.from_char_string('yzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzex', coeff=1)

# time the multiplication (averaged over 100000 repeats)
# print(timeit.timeit('PA.dot(PB)', globals=globals(), number=100000))


print("Single qubit operator tests")
print()

Id = NQO.PauliStr.Ident(1)
pX = NQO.PauliStr.Xop()
pY = NQO.PauliStr.Yop()
pZ = NQO.PauliStr.Zop()



print("Checking idempotency: ")
Idsq = NQO.PauliStr.product(Id,Id)
Idsq.print_string()
Xsq = NQO.PauliStr.product(pX,pX)
Xsq.print_string()
Ysq = NQO.PauliStr.product(pY,pY)
Ysq.print_string()
Zsq = NQO.PauliStr.product(pZ,pZ)
Zsq.print_string()
print()



print("Checking special product -i X Y Z = identity: ")
ThreeThing1 = NQO.PauliStr.product(pX,pY)
ThreeThing1.dot(pZ)
ThreeThing1.rescale(-1j)
ThreeThing1.print_string()

ThreeThing2 = NQO.PauliStr.product(pY,pZ)
ThreeThing2.apply(pX)
ThreeThing2.dot(Id)
ThreeThing2.rescale(-1j)
ThreeThing2.print_string()
print()



print("Checking products of nontrivial Paulis: ")

print("X*Y")
newguy = NQO.PauliStr.product(pX,pY)
newguy.print_string()

print("Y*X")
newguy = NQO.PauliStr.product(pY,pX)
newguy.print_string()

print("X*Z")
newguy = NQO.PauliStr.product(pX,pZ)
newguy.print_string()

print("Z*X")
newguy = NQO.PauliStr.product(pZ,pX)
newguy.print_string()

print("Y*Z")
newguy = NQO.PauliStr.product(pY,pZ)
newguy.print_string()

print("Z*Y")
newguy = NQO.PauliStr.product(pZ,pY)
newguy.print_string()

print()



print("Checking `check_comm' functions for trivial commutators:")
IIcomm = NQO.PauliStr.commutator(Id,Id)
IIcomm.print_string()
XXcomm = NQO.PauliStr.commutator(pX,pX)
XXcomm.print_string()
YYcomm = NQO.PauliStr.commutator(pY,pY)
YYcomm.print_string()
ZZcomm = NQO.PauliStr.commutator(pZ,pZ)
ZZcomm.print_string()
IXcomm = NQO.PauliStr.commutator(Id,pX)
IXcomm.print_string()
IYcomm = NQO.PauliStr.commutator(Id,pY)
IYcomm.print_string()
IZcomm = NQO.PauliStr.commutator(Id,pZ)
IZcomm.print_string()
print()



print("Checking commutators of nontrivial Paulis")
XYcomm = NQO.PauliStr.commutator(pX,pY)
print("[X,Y]")
XYcomm.print_string()

YXcomm = NQO.PauliStr.commutator(pY,pX)
print("[Y,X]")
YXcomm.print_string()

ZXcomm = NQO.PauliStr.commutator(pZ,pX)
print("[Z,X]")
ZXcomm.print_string()

XZcomm = NQO.PauliStr.commutator(pX,pZ)
print("[X,Z]")
XZcomm.print_string()

YZcomm = NQO.PauliStr.commutator(pY,pZ)
print("[Y,Z]")
YZcomm.print_string()

ZYcomm = NQO.PauliStr.commutator(pZ,pY)
print("[Z,Y]")
ZYcomm.print_string()

print()



print("checking ANTIcommutators of Paulis with themselves: ")
IIacomm = NQO.PauliStr.anticommutator(Id,Id)
IIacomm.print_string()

XXacomm = NQO.PauliStr.anticommutator(pX,pX)
XXacomm.print_string()

YYacomm = NQO.PauliStr.anticommutator(pY,pY)
YYacomm.print_string()

ZZacomm = NQO.PauliStr.anticommutator(pZ,pZ)
ZZacomm.print_string()
print()



print("checking ANTIcommutators of Paulis with others (no identity): ")
print("{X,Y}")
XYacomm = NQO.PauliStr.anticommutator(pX,pY.copy())
XYacomm.print_string()

print("{Z,X}")
ZXacomm = NQO.PauliStr.anticommutator(pZ,pX)
ZXacomm.print_string()

print("{Y,Z}")
YZacomm = NQO.PauliStr.anticommutator(pY,pZ)
YZacomm.print_string()
print()

