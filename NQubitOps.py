#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020
@author: Aaron + Ollie
Disclaimer: For internal / private use only; please do not share without permission.
"""

import numpy as np
from sys import exit


####### Pauli String Object
##############################################################################
##############################################################################
class PauliStr():
    """
    Pauli `string' object: Represents a single basis string operator of the
    form O[0] ... O[j] O[j+1] ... O[N-1], where the operator O[j] acts on spin
    "j", and is one of the following operators: Id, Pauli_1 Pauli_2, Pauli_3
    (or Id, X, Y, Z)
    
    The Pauli String is always a Kronecker product operator (over sites).
    
    Pauli strings form an orthonormal basis for the 4**N unique operators
    acting on N qubits.
    
    Each Pauli operator, O[j], is represented by two bits, corresponding to X
    and Z:
    
    Id = '00', X = '10', Z = '01', XZ = '11' = - 1j Y
    
    In order to track Ys, we must account for an overall phase, stored in the
    string's coefficient; by convention, X acts to the left of Z on a given
    site.
    
    Attributes
    ----------
    N : int
        the total number of sites in the chain
    coeff : complex
        a Complex number, the coefficient multiplying the operator string
    X_arr : numpy array, dtype=bool
        a boolean array, giving the sites on which an "X" acts ("1" for an X)
    Z_arr : numpy array, dtype=bool
        a boolean array, giving the sites on which an "Z" acts ("1" for a Z)
        
    """
    
    ### The Pauli matrices (and identity)
    P0_array = np.eye(2)
    P1_array = np.array([[0, 1],  [1, 0]])
    P2_array = np.array([[0, -1j],[1j, 0]])
    P3_array = np.array([[1, 0],  [0, -1]])
    
    ### characters to ignore when initializing from a character string
    ignored_chars = [",", " ", "_", ";"]
    
    def __init__(self, N = 10, coeff = 1.0 + 0.0j, Xs = None, Zs = None):
        """
        Create Pauli string object from boolean arrays representing the X and
        Z operator content
        
        Parameters
        ----------
        N : int, optional
            An integer representing the number of sites upon which the Pauli
            string acts
        coeff : complex, optional
            Overall coefficient multiplying the string. The default is 1.0.
        Xs : numpy array, dtype=bool, optional
            Boolean array, giving the sites on which an "X" acts ("1" for an
            X), from which the string is constructed. Default is all 0's
            (all identities).
        Zs : numpy array, dtype=bool, optional
            Boolean array, giving the sites on which an "Z" acts ("1" for an
            Z), from which the string is constructed. Default is all 0's
            (all identities).
        """
        
        # length of Pauli string / system
        self.N = N                          
        
        # where are the Xs
        if Xs is None:
            self.X_arr = np.zeros(N, dtype='bool')
        elif len(Xs) != N:
            exit("mismatch between X array and number of qubits, N")
        else:
            self.X_arr = Xs
        
        # where are the Zs
        if Zs is None:
            self.Z_arr = np.zeros(N, dtype='bool')
        elif len(Zs) != N:
            exit("mismatch between Z array and number of qubits, N")
        else:
            self.Z_arr = Zs
        
        # overall magnitude and  phase
        self.coeff = coeff                  


    ### Alternate instantiation:
    @classmethod
    def from_char_string(cls, charstr, coeff=1.0 + 0.0j, left_pad=0, right_pad=0):
        """
        Convert a string of characters 1,x,X ; 2,y,Y ; 3,z,Z to boolean
        arrays. Other characters give identity. Ignores characters in
        "ignored_chars"
        
        
        Parameters
        ----------
        cls : PauliString Object
            Create a single basis string
        charstr : string
            String of characters: x,X,1 for (1); y,Y,2 for (2); z,Z,3 for (3);
            all others to Identity
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
        
        for s in PauliStr.ignored_chars:
            charstr.replace(s, "")
        
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
    
    
    @classmethod
    def null_str(cls, N):
        """
        Creates a null operator (the 0) acting on N qubits

        Parameters
        ----------
        N : int
            Number of qubits

        Returns
        -------
        PauliStr object
            A PauliString that acts as the identity with coefficient 0.

        """
        return cls(N, 0.0)

    def dot(self, PStr_B):
        """
        Multiply a PauliString object instance IN PLACE from the RIGHT by
        PStr_B, i.e., A.dot(B) = A*B
        
        Parameters
        ----------
        PStr_B : PauliStr object
            The Pauli string that is being multiplied (from the right) onto
            the current string
        """

        assert(self.N == PStr_B.N)

        self.X_arr = np.logical_xor(self.X_arr, PStr_B.X_arr)
        self.Z_arr = np.logical_xor(self.Z_arr, PStr_B.Z_arr)

        flip = bool(np.sum(np.logical_and(self.Z_arr, PStr_B.X_arr)) % 2)
        if flip:
            self.coeff *= -PStr_B.coeff
        else:
            self.coeff *= PStr_B.coeff 


    def apply(self, PStr_B):
        """
        Multiply a PauliString object instance IN PLACE from the LEFT by
        PStr_B, i.e., A.apply(B) = B*A
        
        Parameters
        ----------
        PStr_B : PauliStr object
            The Pauli string that is being multiplied (from the left) onto the
            current string
        """

        assert(self.N == PStr_B.N)

        self.X_arr = np.logical_xor(self.X_arr, PStr_B.X_arr)
        self.Z_arr = np.logical_xor(self.Z_arr, PStr_B.Z_arr)

        flip = bool(np.sum(np.logical_and(self.X_arr, PStr_B.Z_arr)) % 2)
        if flip:
            self.coeff *= -PStr_B.coeff
        else:
            self.coeff *= PStr_B.coeff
        
            
    def rescale(self, scale):
        """
        Multiply a PauliString object instance IN PLACE by constant factor, scale
        This affects only the "coeff" part of the string
        

        Parameters
        ----------
        scale : complex float
            factor by which the Pauli string is to be multiplied
        """

        self.coeff *= scale


    def print_string(self):
        """
        Print the content of the Pauli string in human readable format
        P_0 -> "e"
        P_1 -> "x"
        P_2 -> "y"
        P_3 -> "z"
        The overall magnitude and phase of the string is printed along with
        its operator content.
        """
        out = ""
        prefactor = 1.0

        for xs, zs in zip(self.X_arr, self.Z_arr):
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

        print("Pauli string: ", self.coeff*prefactor, out)
    
    
    def copy(self):
        """
        Returns a copy of the current object instance. I think this works?

        Returns
        -------
        PauliStr object
            A new instance of the PauliStr class object with identical
            attributes as "self" 

        """
        return PauliStr(self.N, self.coeff, self.X_arr, self.Z_arr)


    ### Static methods (take instances of the class as arguments...NOT in place)

    @staticmethod
    def product(PStr_A, PStr_B):
        """
        Create a NEW PauliString object instance as the product, A*B
        
        Parameters
        ----------
        PStr_A : PauliStr object
            The left Pauli string in the product
        PStr_B : PauliStr object
            The right Pauli string in the product
        
        Returns
        -------
        output : PauliStr object
            The PauliStr object C = A*B
        """

        assert(PStr_A.N == PStr_B.N)

        new_X_arr = np.logical_xor(PStr_A.X_arr, PStr_B.X_arr)
        new_Z_arr = np.logical_xor(PStr_A.Z_arr, PStr_B.Z_arr)
        
        flip = bool(np.sum(np.logical_and(PStr_A.Z_arr, PStr_B.X_arr)) % 2)
        if flip:
            new_coeff = -PStr_A.coeff * PStr_B.coeff
        else:
            new_coeff = PStr_A.coeff * PStr_B.coeff 

        return PauliStr(PStr_A.N, new_coeff, new_X_arr, new_Z_arr)


    @staticmethod
    def check_comm(PStr_A, PStr_B):
        
        assert(PStr_A.N == PStr_B.N)
        
        sign1 = bool(np.sum(np.logical_and(PStr_A.X_arr, PStr_B.Z_arr)) % 2)
        sign2 = bool(np.sum(np.logical_and(PStr_A.Z_arr, PStr_B.X_arr)) % 2)
        
        if sign1 == sign2:
            return True
        else:
            return False
    
    @staticmethod
    def check_anticomm(PStr_A, PStr_B):
        
        assert(PStr_A.N == PStr_B.N)
        
        sign1 = bool(np.sum(np.logical_and(PStr_A.X_arr, PStr_B.Z_arr)) % 2)
        sign2 = bool(np.sum(np.logical_and(PStr_A.Z_arr, PStr_B.X_arr)) % 2)
        
        if sign1 == sign2:
            return False
        else:
            return True
    
    
    @staticmethod
    def commutator(PStr_A, PStr_B):
        assert(PStr_A.N == PStr_B.N)
        
        if not PauliStr.check_comm(PStr_A, PStr_B):
            return (PauliStr.product(PStr_A, PStr_B)).rescale(2.0)
        else:
            return PauliStr.null_str(PStr_A.N)
        
        
    @staticmethod
    def anticommutator(PStr_A, PStr_B):
        assert(PStr_A.N == PStr_B.N)
        
        if not PauliStr.check_anticomm(PStr_A, PStr_B):
            return (PauliStr.product(PStr_A, PStr_B)).rescale(2.0)
        else:
            return PauliStr.null_str(PStr_A.N)

    
    
    @staticmethod
    def old_commutator(PStr_A, PStr_B, anti = False):
        """
        Deprecated method, scheduled for removal
        
        Create a new PauliStr object instance as 
        (i) the commutator, [ A , B ], of A and B (DEFAULT, anti==False) or
        (ii) the ANTI-commutator, { A , B }, of A and B (if anti==True)
        
        Note: Every Pauli string either commutes or anticommutes
        
        Parameters
        ----------
        PStr_A : PauliStr object
            The left (first) Pauli string in the commutator
        PStr_B : PauliStr object
            The rightt (second) Pauli string in the commutator
            
        Note: The ordering is unimportant if anti==True
            
            
        Returns
        -------
        output : PauliStr object
            The PaluiStr object C = [A,B] (commutator) or
            C = {A,B} (anticommutator)
        
        OR
        
        output : None
            If the [anti]commutator is zero
        """

        assert(PStr_A.N == PStr_B.N)
        
        new_X_arr = np.logical_xor(PStr_A.X_arr, PStr_B.X_arr)
        new_Z_arr = np.logical_xor(PStr_A.Z_arr, PStr_B.Z_arr)
        
        sign1 = bool(np.sum(np.logical_and(PStr_A.X_arr, PStr_B.Z_arr)) % 2)
        sign2 = bool(np.sum(np.logical_and(PStr_A.Z_arr, PStr_B.X_arr)) % 2)
        
        if anti:
            if sign1 != sign2:
                return None
            else:
                if sign1:
                    new_coeff = -2.0 * PStr_A.coeff * PStr_B.coeff
                else:
                    new_coeff = 2.0 * PStr_A.coeff * PStr_B.coeff
        else:
            if sign1 == sign2:
                return None
            else:
                if sign1:
                    new_coeff = -2.0 * PStr_A.coeff * PStr_B.coeff
                else:
                    new_coeff = 2.0 * PStr_A.coeff * PStr_B.coeff
            
        
        return PauliStr(PStr_A.N, new_coeff, new_X_arr, new_Z_arr)
