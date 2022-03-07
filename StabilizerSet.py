#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020

@author: Aaron + Ollie

Disclaimer: For internal / private use only; please do not share without my permission.
"""
import numpy as np
import sympy
import random
import NQubitOps as NQO

####### Stabilizer Set Object
##############################################################################
##############################################################################
class StabilizerSet():
    """
    Represents a set of basis string operator of the form
    O[0] ... O[j] O[j+1] ... O[N-1]. The Pauli Strings are always a Kronecker
    product operator (over sites). The set of Pauli strings is of length N.
    It therefore uniquely defines a state |psi> via projection.

    The operator O[j] acts on spin "j", and is one of the following operators:

        Id, Pauli_1 Pauli_2, Pauli_3 (or Id, X, Y, Z)
    
    Each Pauli is represented by two bits:

        00 = Id, 10 = X, 11 = XZ, 01 = Z 

    Note: since 11 represents XZ = -iY, there is an associated phase, stored
    in the overall coefficient for the Pauli string
    
    Attributes
    ----------
    N : int
        the total number of sites in the chain, and also the number of
        independent Pauli strings. The strings must be mutually commuting
        and independent.
    coeff : complex numpy array
        an array of complex numbers, corresponding to the coefficient
        multiplying the operator string
    X_arr : 2D numpy array, dtype=bool
        a boolean array, giving the sites on which an "X" acts ("1" for an X).
        First index specifies the string, second index specifies the site.
    Z_arr : 2D numpy array, dtype=bool
        a boolean array, giving the sites on which an "Z" acts ("1" for a Z).
        First index specifies the string, second index specifies the site.
    """

    P0_array = np.eye(2)
    P1_array = np.array([[0, 1],  [1, 0]])
    P2_array = np.array([[0, -1j],[1j, 0]])
    P3_array = np.array([[1, 0],  [0, -1]])


    XX_dict = {}
    XX_dict[(0, 0, 0, 0)] = [NQO.PauliStr.from_XZ_string('00;00', 1),   NQO.PauliStr.from_XZ_string('00;00', +1), NQO.PauliStr.from_XZ_string('00;00', +1), NQO.PauliStr.from_XZ_string('00;00', +1)]
    XX_dict[(1, 0, 0, 0)] = [NQO.PauliStr.from_XZ_string('10;00', 1),   NQO.PauliStr.from_XZ_string('10;00', +1), NQO.PauliStr.from_XZ_string('10;00', +1), NQO.PauliStr.from_XZ_string('10;00', +1)]
    XX_dict[(0, 1, 0, 0)] = [NQO.PauliStr.from_XZ_string('01;00', 1),   NQO.PauliStr.from_XZ_string('01;00', -1), NQO.PauliStr.from_XZ_string('11;10', -1), NQO.PauliStr.from_XZ_string('11;10', +1)]
    XX_dict[(1, 1, 0, 0)] = [NQO.PauliStr.from_XZ_string('11;00', 1),   NQO.PauliStr.from_XZ_string('11;00', -1), NQO.PauliStr.from_XZ_string('01;10', -1), NQO.PauliStr.from_XZ_string('01;10', +1)]

    XX_dict[(0, 0, 1, 0)] = [NQO.PauliStr.from_XZ_string('00;10', 1),   NQO.PauliStr.from_XZ_string('00;10', +1), NQO.PauliStr.from_XZ_string('00;10', +1), NQO.PauliStr.from_XZ_string('00;10', +1)]
    XX_dict[(1, 0, 1, 0)] = [NQO.PauliStr.from_XZ_string('10;10', 1),   NQO.PauliStr.from_XZ_string('10;10', +1), NQO.PauliStr.from_XZ_string('10;10', +1), NQO.PauliStr.from_XZ_string('10;10', +1)]
    XX_dict[(0, 1, 1, 0)] = [NQO.PauliStr.from_XZ_string('01;10', 1),   NQO.PauliStr.from_XZ_string('01;10', -1), NQO.PauliStr.from_XZ_string('11;00', -1), NQO.PauliStr.from_XZ_string('11;00', +1)]
    XX_dict[(1, 1, 1, 0)] = [NQO.PauliStr.from_XZ_string('11;10', 1),   NQO.PauliStr.from_XZ_string('11;10', -1), NQO.PauliStr.from_XZ_string('01;00', -1), NQO.PauliStr.from_XZ_string('01;00', +1)]

    XX_dict[(0, 0, 0, 1)] = [NQO.PauliStr.from_XZ_string('00;01', 1),   NQO.PauliStr.from_XZ_string('00;01', -1), NQO.PauliStr.from_XZ_string('10;11', -1), NQO.PauliStr.from_XZ_string('10;11', +1)]
    XX_dict[(1, 0, 0, 1)] = [NQO.PauliStr.from_XZ_string('10;01', 1),   NQO.PauliStr.from_XZ_string('10;01', -1), NQO.PauliStr.from_XZ_string('00;11', -1), NQO.PauliStr.from_XZ_string('00;11', +1)]
    XX_dict[(0, 1, 0, 1)] = [NQO.PauliStr.from_XZ_string('01;01', 1),   NQO.PauliStr.from_XZ_string('01;01', +1), NQO.PauliStr.from_XZ_string('01;01', -1), NQO.PauliStr.from_XZ_string('01;01', -1)]
    XX_dict[(1, 1, 0, 1)] = [NQO.PauliStr.from_XZ_string('11;01', 1),   NQO.PauliStr.from_XZ_string('11;01', +1), NQO.PauliStr.from_XZ_string('11;01', -1), NQO.PauliStr.from_XZ_string('11;01', -1)]

    XX_dict[(0, 0, 1, 1)] = [NQO.PauliStr.from_XZ_string('00;11', 1),   NQO.PauliStr.from_XZ_string('00;11', -1), NQO.PauliStr.from_XZ_string('10;01', -1), NQO.PauliStr.from_XZ_string('10;01', +1)]
    XX_dict[(1, 0, 1, 1)] = [NQO.PauliStr.from_XZ_string('10;11', 1),   NQO.PauliStr.from_XZ_string('10;11', -1), NQO.PauliStr.from_XZ_string('00;01', -1), NQO.PauliStr.from_XZ_string('00;01', +1)]
    XX_dict[(0, 1, 1, 1)] = [NQO.PauliStr.from_XZ_string('01;11', 1),   NQO.PauliStr.from_XZ_string('01;11', +1), NQO.PauliStr.from_XZ_string('01;11', -1), NQO.PauliStr.from_XZ_string('01;11', -1)]
    XX_dict[(1, 1, 1, 1)] = [NQO.PauliStr.from_XZ_string('11;11', 1),   NQO.PauliStr.from_XZ_string('11;11', +1), NQO.PauliStr.from_XZ_string('11;11', -1), NQO.PauliStr.from_XZ_string('11;11', -1)]


    def __init__(self, N=10, coeff=None, Xs=None, Zs=None):
        """
        Create Stabilizer object from boolean arrays representing the X and
        Z operator content in each string

        Parameters
        ----------
        N : int, optional
            An integer representing the number of sites upon which the Pauli
            string acts. Also equals the number of stabilizers.
        coeff : complex numpy array, optional
            Overall coefficient multiplying the string. The default is 1.0.
        Xs : 2D numpy array, dtype=bool, optional
            Boolean array, giving the sites on which an "X" acts ("1" for an
            X), from which the string is constructed. Default is all 0's (all
            identities).
        Zs : 2D numpy array, dtype=bool, optional
            Boolean array, giving the sites on which an "Z" acts ("1" for an
            Z), from which the string is constructed. Default is all 0's (all
            identities).
        """

        self.N = N                              # length of Pauli string

        if (Xs is not None) and (Zs is not None):
            self.X_arr = Xs
            self.Z_arr = Zs
        else:
            self.X_arr = np.zeros((N, N), dtype='bool')
            self.Z_arr = np.zeros((N, N), dtype='bool')

        if coeff is not None:
            self.coeff = coeff                  # overall magnitude & phase
        else:
            self.coeff = np.ones(N, dtype='complex')


    def set_string(self, index, charstr, coeff=1.0 + 0.0j, left_pad=0):
        """
        Create Pauli string corresponding to the stabilizer in the row
        specified by "index".

        Convert a string of characters 1,x,X ; 2,y,Y ; 3,z,Z to boolean
        array. Other characters give identity.

        Parameters
        ----------
        index : integer
            row index of string to be set
        charstr : string
            String of characters: x,X,1 for (1); y,Y,2 for (2); z,Z,3 for (3);
            all others to Identity
        coeff : complex, optional
            Overall coefficient (complex) for the string. The default is 1.0.
        left_pad : int, optional
            Number of identities to pad to the LEFT of the character string
        right_pad : int, optional
            Number of identities to pad to the RIGHT of the character string
        """

        assert(left_pad >= 0)
        assert(len(charstr) + left_pad <= self.N)

        right_pad = self.N - len(charstr) - left_pad # (right_pad >= 0)

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

        self.X_arr[index, :] = X_int
        self.Z_arr[index, :] = Z_int


    def dot(self, PStr_B):
        """
        Multiply a StabilizerSet Instance IN PLACE from the right by PStr_B,
        i.e.,

            A.dot(B) = A*B

        for *all* strings belonging to the set

        Parameters
        ----------
        PStr_B : PauliString Object Instance
            The Pauli string that is being multiplied (from the right onto our
            current string)
        """

        assert(self.N == SetB.N)

        self.X_arr = np.logical_xor(self.X_arr, PStr_B.X_arr[np.newaxis, :])
        self.Z_arr = np.logical_xor(self.Z_arr, PStr_B.Z_arr[np.newaxis, :])

        sign_arr = np.logical_and(self.Z_arr, PStr_B.X_arr[np.newaxis, :])
        self.coeff = self.coeff * PStr_B.coeff * (-1)**(np.sum(sign_arr, axis=1) % 2)


    def apply(self, PStr_B):
        """
        Multiply a StabilizerSet Instance IN PLACE from the LEFT by PStr_B,
        i.e.,

            A.apply(B) = B*A

        for *all* strings belonging to the set

        Parameters
        ----------
        PStr_B : PauliString Object Instance
            The Pauli string that is being multiplied (from the left onto our
            current string)
        """

        assert(self.N == SetB.N)

        self.X_arr = np.logical_xor(PStr_B.X_arr[np.newaxis, :], self.X_arr)
        self.Z_arr = np.logical_xor(PStr_B.Z_arr[np.newaxis, :], self.Z_arr)

        sign_arr = np.logical_and(PStr_B.Z_arr[np.newaxis, :], self.X_arr)
        self.coeff = self.coeff * PStr_B.coeff * (-1)**(np.sum(sign_arr, axis=1) % 2)


    def dot_single(self, index, PStr_B):
        """
        Multiply a single PauliString Object Instance IN PLACE from the right
        by PStr_B
        
            A.dot(B) = A*B

        Parameters
        ----------
        PStr_B : PauliString Object Instance
            The Pauli string that is being multiplied (from the right onto our
            current string)
        """

        assert(self.N == SetB.N)

        self.X_arr[index, :] = np.logical_xor(self.X_arr[index, :], PStr_B.X_arr)
        self.Z_arr[index, :] = np.logical_xor(self.Z_arr[index, :], PStr_B.Z_arr)

        sign_arr = np.logical_and(self.Z_arr[index, :], PStr_B.X_arr)
        self.coeff[index] = self.coeff[index] * PStr_B.coeff * (-1)**(np.sum(sign_arr) % 2)


    def entanglement_entropy(self, pos=None):
        """
        Find the entanglement entropy of the state specified by the set of
        stabilizers. This works by performing a Gaussian elimination to find
        the left-endpoints in the "clipped gauge". The formula for the
        entanglement entropy is:

            S_A(x) = sum_{y <= x} (rho(y) - 1)

        where rho(y) is the density of left endpoints on site y.

        Parameters
        ----------
        pos : integer, optional
            position of the cut for computing the entanglement entropy.
            The default is to evaluate the half-chain entropy.

        Returns
        -------
        output: float
            value of the entanglement entropy (logarithm base 2)
        """


        # Put all X's and Z's into one array, with X's preceding Z's
        full_bitarr = np.zeros((self.N, 2*self.N), dtype='int')
        full_bitarr[:, ::2] = self.X_arr
        full_bitarr[:, 1::2] = self.Z_arr

        # perform Gaussian elimination to row echelon form (not sure about
        #   speed of this function)
        result = sympy.Matrix(full_bitarr).rref()
        clipped_int = np.array(result[0].tolist(), dtype='int')

        full_bitarr = np.mod(clipped_int, 2)

        # find left end points: argmax returns argument of first True
        first_nonzero = (full_bitarr!=0).argmax(axis=1) // 2

        left_endpoints = np.zeros(self.N, dtype='int')

        #unique, counts = np.unique(first_nonzero, return_counts=True)
        #left_endpoints[unique] += counts
        np.add.at(left_endpoints, first_nonzero, 1)

        if pos is not None:
            assert((pos > 0) and (pos < self.N))
            return np.sum(left_endpoints[:pos]) - pos
        else:
            return np.sum(left_endpoints[:self.N//2]) - self.N//2


    def random_Z_gate(self, pos):
        r"""
        Apply gate of the form U = exp(i h Z) to the state of the system.

        At the level of the stabilizers, this leads to a transformation 

            S -> U^dagger S U

        This leads to the following possiblilties:

            X -> \pm X
        or  X -> \pm XZ
        
        therefore (01) -> (01) always
        while     (10) -> (10) or (11) with a potential sign change    
        """

        rand = random.randint(0, 3) # random integer from {0, 1, 2, 3}

        Z_dict = {}
        Z_dict[(0, 0)] = [NQO.PauliStr.Id(coeff=+1),   NQO.PauliStr.Id(coeff=1),
                          NQO.PauliStr.Id(coeff=+1),   NQO.PauliStr.Id(coeff=1)]
        Z_dict[(1, 0)] = [NQO.PauliStr.XI(coeff=+1),   NQO.PauliStr.XI(coeff=-1),
                          NQO.PauliStr.XZ(coeff=+1),   NQO.PauliStr.XZ(coeff=-1)]
        Z_dict[(0, 1)] = [NQO.PauliStr.IZ(coeff=+1),   NQO.PauliStr.IZ(coeff=1),
                          NQO.PauliStr.IZ(coeff=+1),   NQO.PauliStr.IZ(coeff=1)]
        Z_dict[(1, 1)] = [NQO.PauliStr.XZ(coeff=+1),   NQO.PauliStr.XZ(coeff=-1),
                          NQO.PauliStr.XI(coeff=+1),   NQO.PauliStr.XI(coeff=-1)]

        new_strings = [Z_dict[(self.X_arr[n, pos], self.Z_arr[n, pos])][rand] for n in range(self.N)]

        signs = np.array([new_strings[n].coeff for n in range(self.N)])
        new_X = np.array([new_strings[n].X_arr[0] for n in range(self.N)])
        new_Z = np.array([new_strings[n].Z_arr[0] for n in range(self.N)])

        self.X_arr[:, pos] = new_X
        self.Z_arr[:, pos] = new_Z
        self.coeff *= signs


    def print_stabilizers(self):
        """
        Print the content of the StabilizerSet in human readable format

        P_0 -> "e"
        P_1 -> "x"
        P_2 -> "y"
        P_3 -> "z"

        The overall magnitude and phase of each string is printed along with
        its operator content.
        """

        for i in range(self.N):
            out = ""
            prefactor = 1

            for xs, zs in zip(self.X_arr[i, :], self.Z_arr[i, :]):
                if (xs == 0):
                    if (zs == 0):
                        out += "e"
                    elif (zs == 1):
                        out += "z"
                    else:
                        raise ValueError("Unexpected value in Z_arr: {}, must be either 0 or 1".format(zs))
                elif (xs == 1):
                    if (zs == 0):
                        out += "x"
                    elif (zs == 1):
                        out += "y"
                        prefactor *= (-1j)
                    else:
                        raise ValueError("Unexpected value in Z_arr: {}, must be either 0 or 1".format(zs))
                else:
                    raise ValueError("Unexpected value in X_arr: {}, must be either 0 or 1".format(xs))

            print(self.coeff[i]*prefactor, out)

