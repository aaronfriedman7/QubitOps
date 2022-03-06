#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020

@author: Aaron + Ollie

Disclaimer: For internal / private use only; please do not share without my permission.
"""
import timeit
import numpy as np


####### Pauli String Object
################################################################################################
################################################################################################
class PauliStr():
    """
    Represents a single basis string operator of the form O[0] ... O[j] O[j+1] ... O[N-1]
    The Pauli String is always a Kronecker product operator (over sites)
    The operator O[j] acts on spin "j", and is one of the following operators: Id, Pauli_1 Pauli_2, Pauli_3 (or Id, X, Y, Z)
    
    Each Pauli is represented by two bits

    00 = Id, 10 = X, 11 = XZ, 01 = Z 

    Note: since 11 represents XZ = -iY, there is an associated phase, stored in the overall
    coefficient for the Pauli string
    
    Attributes
    ----------
    N : int
        the total number of sites in the chain
    coeff : complex
        a Complex number, the coefficient multiplying the operator string
    X_int : numpy array, dtype=bool
        a boolean array, giving the sites on which an "X" acts ("1" for an X)
    Z_int : numpy array, dtype=bool
        a boolean array, giving the sites on which an "Z" acts ("1" for a Z)
        
    """
    
    P0_array = np.eye(2)
    P1_array = np.array([[0, 1],  [1, 0]])
    P2_array = np.array([[0, -1j],[1j, 0]])
    P3_array = np.array([[1, 0],  [0, -1]])
    
    def __init__(self, N=10, coeff=1+0j, Xint=None, Zint=None):
        """
        Create Pauli string object from boolean arrays representing the X and Z operator content

        Parameters
        ----------
        N : int, optional
            An integer representing the number of sites upon which the Pauli string acts
        coeff : complex, optional
            Overall coefficient multiplying the string. The default is 1.0.
        Xint : numpy array, dtype=bool, optional
            Boolean array, giving the sites on which an "X" acts ("1" for an X), from which the string is constructed.
            Default is all 0's (all identities).
        Zint : numpy array, dtype=bool, optional
            Boolean array, giving the sites on which an "Z" acts ("1" for an Z), from which the string is constructed.
            Default is all 0's (all identities).
        """

        self.N = N                          # length of Pauli string

        if (Xint is not None) and (Zint is not None):
            self.X_int = Xint
            self.Z_int = Zint
        else:
            self.X_int = np.zeros(N, dtype='bool')
            self.Z_int = np.zeros(N, dtype='bool')

        self.coeff = coeff                  # overall magnitude & phase


    ### Alternate instantiation:
    @classmethod
    def from_char_string(cls, charstr, coeff=1.0 + 0.0j, left_pad=0, right_pad=0):
        """
        Convert a string of characters 1,x,X ; 2,y,Y ; 3,z,Z to boolean arrays. Other characters give identity.

        Parameters
        ----------
        cls : PauliString Object
            Create a single basis string
        charstr : string
            String of characters: x,X,1 for (1); y,Y,2 for (2); z,Z,3 for (3); all others to Identity
        coeff : complex, optional
            Overall coefficient (complex) for the string. The default is 1.0.
        left_pad : int, optional
            Number of identities to pad to the LEFT of the character string
        right_pad : int, optional
            Number of identities to pad to the RIGHT of the character string

        Returns
        -------
        PauliString Class Object
            With the operator content stored as an integer

        """

        N = len(charstr) + left_pad + right_pad

        l_arr = np.zeros(left_pad, dtype='bool')
        r_arr = np.zeros(right_pad, dtype='bool')

        x_arr = []
        z_arr = []

        for char in charstr:
            if char == "1" or char == "x" or char == "X":
                x_arr.append(1)
                z_arr.append(0)
            elif char == "2" or char == "y" or char == "Y":
                x_arr.append(1)
                z_arr.append(1)
                coeff *= 1j
            elif char == "3" or char == "z" or char == "Z":
                x_arr.append(0)
                z_arr.append(1)
            else:
                x_arr.append(0)
                z_arr.append(0)

        x_arr = np.array(x_arr, dtype='bool')
        z_arr = np.array(z_arr, dtype='bool')

        X_int = np.r_[l_arr, x_arr, r_arr]
        Z_int = np.r_[l_arr, z_arr, r_arr]

        return cls(N, coeff, X_int, Z_int)


    def dot(self, PStr_B):
        """
        Multiply a PauliString Object Instance IN PLACE from the right by PStr_B
        A.dot(B) = A*B

        Parameters
        ----------
        PStr_B : PauliString Object Instance
            The Pauli string that is being multiplied (from the right onto our current string)

        Returns
        -------
        output : PauliString Object Instance
            The PauliString Object C = A*B = A.dot(B) 
            Note: A is the original "SELF" string object and B is "Int_Str_B"
        """

        assert(self.N == PStr_B.N)

        self.X_int = np.logical_xor(self.X_int, PStr_B.X_int)
        self.Z_int = np.logical_xor(self.Z_int, PStr_B.Z_int)

        sign_arr = np.logical_and(self.Z_int, PStr_B.X_int)
        self.coeff = self.coeff * PStr_B.coeff * (-1)**(np.sum(sign_arr) % 2)


    def apply(self, PStr_B, loc=0):
        """
        Multiply a PauliString Object Instance IN PLACE from the right by PStr_B
        A.dot(B) = A*B

        The string PStr_B can have a length smaller than or equal to that of the present
        instance. Open boundary conditions are assumed, so that PStr_B does not wrap around
        the 

        Parameters
        ----------
        PStr_B : PauliString Object Instance
            The Pauli string that is being multiplied (from the right onto our current string)
        loc : int, optional
            Leftmost site at which the string PStr_B is to be applied, so that PStr_B spans the sites
            [loc, ..., loc + PStr_B.size - 1]
        """

        assert(self.N >= PStr_B.N)

        left = loc
        right = loc + PStr_B.N 
        assert(right <= self.N)


        self.X_int[left:right] = np.logical_xor(self.X_int[left:right], PStr_B.X_int)
        self.Z_int[left:right] = np.logical_xor(self.Z_int[left:right], PStr_B.Z_int)

        sign_arr = np.logical_and(self.Z_int[left:right], PStr_B.X_int)
        self.coeff = self.coeff * PStr_B.coeff * (-1)**(np.sum(sign_arr) % 2)


    @staticmethod
    def product(PStr_A, PStr_B):
        """
        Create a new PauliString Object Instance as the product, A*B

        Utilizes the following formula for multiplication on a single site:
        
            X^a1 Z^b1 X^a2 Z^b2 = (-1)^(b1 & a2) X^((a1+a2)%2) Z^((b1+b2)%2)

        Parameters
        ----------
        PStr_A : PauliString Object Instance
            The left Pauli string in the product
        PStr_B : PauliString Object Instance
            The right Pauli string in the product
        
        Returns
        -------
        output : PauliString Object Instance
            The PauliString Object C = A*B

        """

        assert(PStr_A.N == PStr_B.N)

        new_X_int = np.logical_xor(PStr_A.X_int, PStr_B.X_int)
        new_Z_int = np.logical_xor(PStr_A.Z_int, PStr_B.Z_int)

        sign_arr = np.logical_and(PStr_A.Z_int, PStr_B.X_int)
        new_coeff = PStr_A.coeff * PStr_B.coeff * (-1)**(np.sum(sign_arr) % 2)

        return PauliStr(PStr_A.N, new_coeff, new_X_int, new_Z_int)


    @staticmethod
    def commutator(PStr_A, PStr_B, anti=False):
        """
        Create a new PauliString Object Instance as the commutator (Default), [ A , B ], of two PauliStrings
        OR the ANTI-commutator (anti==True), { A , B }, of two IntStrings
        The ordering is unimportant if anti==True.
        NOTE: every Pauli string either commutes or anticommutes


        Parameters
        ----------
        PStr_A : PauliString Object Instance
            The left Pauli string in the product
        PStr_B : PauliString Object Instance
            The right Pauli string in the product
            
            
        Returns
        -------
        output : PauliString Object Instance
            The PaluiString Object C = A B +/- B A
        
        OR
        
        output : None
            If the [anti]commutator is zero

        """

        assert(PStr_A.N == PStr_B.N)

        AB_sign_arr = np.logical_and(PStr_A.Z_int, PStr_B.X_int)
        AB_sign = (-1)**(np.sum(AB_sign_arr) % 2)

        BA_sign_arr = np.logical_and(PStr_B.Z_int, PStr_A.X_int)
        BA_sign = (-1)**(np.sum(BA_sign_arr) % 2)

        # determine whether the two operator orderings have opposite sign
        if AB_sign*BA_sign == 1:
            isCommuting = True
        else:
            isCommuting = False

        # if commute, and computing commutator, return None, else return anticommutator
        if isCommuting:
            if anti:
                new_coeff = PStr_A.coeff * PStr_B.coeff * (AB_sign + BA_sign)
                new_X_int = np.logical_xor(PStr_A.X_int, PStr_B.X_int)
                new_Z_int = np.logical_xor(PStr_A.Z_int, PStr_B.Z_int)

                return PauliStr(PStr_A.N, new_coeff, new_X_int, new_Z_int)

            else:
                return None
        # if anticommute and computing anticommutator, return None, else return commutator
        else:
            if anti:
                return None
            else:
                new_coeff = PStr_A.coeff * PStr_B.coeff * (AB_sign - BA_sign)
                new_X_int = np.logical_xor(PStr_A.X_int, PStr_B.X_int)
                new_Z_int = np.logical_xor(PStr_A.Z_int, PStr_B.Z_int)

                return PauliStr(PStr_A.N, new_coeff, new_X_int, new_Z_int)



    def print_string(self):
        """
        Print the content of the Pauli string in human readable format

        P_0 -> "e"
        P_1 -> "x"
        P_2 -> "y"
        P_3 -> "z"

        The overall magnitude and phase of the string is printed along with its operator
        content.
        """
        out = ""
        prefactor = 1

        for xs, zs in zip(self.X_int, self.Z_int):
            if (xs == 0):
                if (zs == 0):
                    out += "e"
                elif (zs == 1):
                    out += "z"
                else:
                    raise ValueError("Unexpected value in Z_int: {}, must be either 0 or 1".format(zs))
            elif (xs == 1):
                if (zs == 0):
                    out += "x"
                elif (zs == 1):
                    out += "y"
                    prefactor *= (-1j)
                else:
                    raise ValueError("Unexpected value in Z_int: {}, must be either 0 or 1".format(zs))
            else:
                raise ValueError("Unexpected value in X_int: {}, must be either 0 or 1".format(xs))

        print("Pauli string: ", self.coeff*prefactor, out)



# create two long strings
PA = PauliStr.from_char_string('xyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyze', coeff=1)
PB = PauliStr.from_char_string('yzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzexyzex', coeff=1)

# time the multiplication (averaged over 100000 repeats)
print(timeit.timeit('PA.dot(PB)', globals=globals(), number=100000))
