#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020
@author: Aaron + Ollie
Disclaimer: For internal / private use only; please do not share without permission.
"""

import numpy as np
import scipy.sparse as ss


### The Pauli matrices (and identity) as DENSE numpy arrays (ndarray)
P0_array = np.eye(2)
P1_array = np.array([[0, 1],  [1, 0]])
P2_array = np.array([[0, -1j],[1j, 0]])
P3_array = np.array([[1, 0],  [0, -1]])

Dense_Paulis = [P0_array, P1_array, P2_array, P3_array]

### The Pauli matrices (and identity) as SPARSE scipy arrays (scipy.sparse.csr_matrix)
P0_sparse = ss.csr_matrix((np.array([1.0,1.0]),(np.array([0,1]),np.array([0,1]))), shape=(2,2), dtype='complex')
P1_sparse = ss.csr_matrix((np.array([1.0,1.0]),(np.array([0,1]),np.array([1,0]))), shape=(2,2), dtype='complex')
P2_sparse = ss.csr_matrix((np.array([-1.0j,1.0j]),(np.array([0,1]),np.array([1,0]))), shape=(2,2), dtype='complex')
P3_sparse = ss.csr_matrix((np.array([1.0,-1.0]),(np.array([0,1]),np.array([0,1]))), shape=(2,2), dtype='complex')

Sparse_Paulis = [P0_sparse, P1_sparse, P2_sparse, P3_sparse]


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

    PauliIntDict = {(False,False): 0, (True,False): 1, (True,True): 2, (False, True): 3}
    
    PauliStrDict = {(False,False): 'e', (True,False): 'x', (True,True): 'y', (False, True): 'z'}
    
    
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
            raise ValueError("Mismatch between N={} and length of X array = {}".format(N,len(Xs)))
        else:
            self.X_arr = Xs
        
        # where are the Zs
        if Zs is None:
            self.Z_arr = np.zeros(N, dtype='bool')
        elif len(Zs) != N:
            raise ValueError("Mismatch between N={} and length of Z array = {}".format(N,len(Zs)))
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
            charstr = charstr.replace(s, "")
        
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
    
    ### Alternate instantiation:
    @classmethod
    def clone(cls, PStr):
        """
        Instantiation as a clone of existing PauliString

        Parameters
        ----------
        PStr : PauliStr
            The PauliStr object to be cloned.

        Returns
        -------
        PauliStr
            New PauliStr object with same properties as PStr.

        """
        return cls(PStr.N, PStr.coeff, PStr.X_arr, PStr.Z_arr)
    
    ### Alternate instantiation:
    @classmethod
    def from_XZ_string(cls, xz_str, coeff=1.0 + 0.0j, left_pad=0, right_pad=0):
        """
        Convert a string of characters 0, 1 to boolean arrays. 
        
        
        Parameters
        ----------
        cls : PauliString Object
            Create a single basis string
        xz_str : string
            String of characters: even elements correspond to elements of
            X_arr, odd elements correspond to elements of Z_arr. For example,
            '0110' becomes Z X on adjacent sites. Length of string is twice
            number of sites.
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
            xz_str = xz_str.replace(s, "")
        

        assert((len(xz_str) % 2) == 0)
        N = len(xz_str)//2 + left_pad + right_pad

        l_arr = np.zeros(left_pad, dtype='bool')
        r_arr = np.zeros(right_pad, dtype='bool')

        x_arr = []
        z_arr = []

        for char in xz_str[::2]:
            if char == "1":
                x_arr.append(1)
            elif char == "0":
                x_arr.append(0)
            else:
                raise ValueError("Encountered invalid character {} in string".format(char))

        for char in xz_str[1::2]:
            if char == "1":
                z_arr.append(1)
            elif char == "0":
                z_arr.append(0)
            else:
                raise ValueError("Encountered invalid character {} in string".format(char))


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

    @classmethod
    def Ident(cls, N=10, coeff=1.0+0.0j):
        """
        Creates an identity operator on N sites

        Parameters
        ----------
        N : int, optional
            Number of qubits in the chain. The default is 10.
        coeff : complex, optional
            Coefficient multiplying the identity. The default is 1.

        Returns
        -------
        PauliStr object
            A PauliString that acts as the identity on N sites, with some coefficient.

        """
        return cls(N,coeff,np.zeros(N, dtype='bool'),np.zeros(N, dtype='bool'))
    

    @classmethod
    def Xop(cls, coeff=1.0+0.0*1j):
        """ Single site X operator, multiplied by coeff """
        return cls(1, coeff, np.array([1]), np.array([0]))


    @classmethod
    def Zop(cls, coeff=1.0+0.0*1j):
        """ Single site Z operator, multiplied by coeff """
        return cls(1, coeff, np.array([0]), np.array([1]))


    @classmethod
    def Yop(cls, coeff=1.0+0.0*1j):
        """ Single site Y operator, multiplied by coeff"""
        return cls(1, 1j*coeff, np.array([1]), np.array([1]))



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
        
        flip = bool(np.sum(np.logical_and(self.Z_arr, PStr_B.X_arr)) % 2)
        if flip:
            self.coeff *= -PStr_B.coeff
        else:
            self.coeff *= PStr_B.coeff 

        self.X_arr = np.logical_xor(self.X_arr, PStr_B.X_arr)
        self.Z_arr = np.logical_xor(self.Z_arr, PStr_B.Z_arr)


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
        
        flip = bool(np.sum(np.logical_and(self.X_arr, PStr_B.Z_arr)) % 2)
        if flip:
            self.coeff *= -PStr_B.coeff
        else:
            self.coeff *= PStr_B.coeff

        self.X_arr = np.logical_xor(self.X_arr, PStr_B.X_arr)
        self.Z_arr = np.logical_xor(self.Z_arr, PStr_B.Z_arr)

            
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


### We might want to improve the print string function?
    
    def where_Id(self):
        NxORz = np.logical_not(np.logical_or(self.X_arr,self.Z_arr))
        return np.nonzero(NxORz)
    
    def where_X(self):
        xNOTz = np.logical_and(self.X_arr,np.logical_not(self.Z_arr))
        return np.nonzero(xNOTz)
    
    def where_Y(self):
        xANDz = np.logical_and(self.X_arr,self.Z_arr)
        return np.nonzero(xANDz)
    
    def where_Z(self):
        zNOTx = np.logical_and(self.Z_arr,np.logical_not(self.X_arr))
        return np.nonzero(zNOTx)
    
    
### note zip(A,B) is an iterable
### [foo for foo in zip(A,B)] = [(a_0,b_0), (a_1,b_1), ... (a_{n-1} , b_{n-1})]
    
    @property
    def char_list(self):
        return [PauliStr.PauliStrDict[guy] for guy in zip(self.X_arr,self.Z_arr)]
    
    @property
    def int_list(self):
        return [PauliStr.PauliIntDict[guy] for guy in zip(self.X_arr,self.Z_arr)]
    
    
    @property
    def char_string(self):
        prefactor = self.coeff
        PauliOrder = [PauliStr.PauliStrDict[guy] for guy in zip(self.X_arr,self.Z_arr)]
        out = "".join(PauliOrder)
        numys = out.count("y")
        prefactor *= ((-1j)**numys)
        return (prefactor, out)
    
    @property
    def int_string(self):
        prefactor = self.coeff
        PauliOrder = [str(PauliStr.PauliIntDict[guy]) for guy in zip(self.X_arr,self.Z_arr)]
        out = "".join(PauliOrder)
        numys = out.count("2")
        prefactor *= ((-1j)**numys)
        return (prefactor, out)
    
   
    def print_string(self):
        prefactor = self.coeff
        PauliOrder = [PauliStr.PauliStrDict[guy] for guy in zip(self.X_arr,self.Z_arr)]
        out = "".join(PauliOrder)
        prefactor *= ((-1j)**(out.count("y")))
        print("Pauli string: ", prefactor, out)
    
    
    def to_ndarray(self):
        ints = self.int_list
        prefactor = self.coeff
        numys = ints.count(2)
        prefactor *= ((-1j)**numys)
        assert len(ints) == self.N
        
        Op = Dense_Paulis[ints[0]]
        for foo in range(1,self.N):
            nextguy = Dense_Paulis[ints[foo]]
            Op = np.kron(Op,nextguy)
        
        assert np.shape(Op) == (2**self.N,2**self.N)
        
        return prefactor*Op
        

    def to_sparse(self):
        ints = self.int_list
        prefactor = self.coeff
        numys = ints.count(2)
        prefactor *= ((-1j)**numys)
        assert len(ints) == self.N
        
        Op = Sparse_Paulis[ints[0]]
        for foo in range(1,self.N):
            nextguy = Sparse_Paulis[ints[foo]]
            Op = ss.kron(Op,nextguy)
        Op.multiply(prefactor)
        
        return Op
    
    
    
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
        
        flip = bool(np.sum(np.logical_and(PStr_A.Z_arr, PStr_B.X_arr)) % 2)
        
        if flip:
            new_coeff = -PStr_A.coeff * PStr_B.coeff
        else:
            new_coeff = PStr_A.coeff * PStr_B.coeff

        new_X_arr = np.logical_xor(PStr_A.X_arr, PStr_B.X_arr)
        new_Z_arr = np.logical_xor(PStr_A.Z_arr, PStr_B.Z_arr)
        
        C = PauliStr(PStr_A.N, new_coeff, new_X_arr, new_Z_arr)

        return C


    @staticmethod
    def check_comm(PStr_A, PStr_B):
        """
        Checks if the two Pauli strings commute. Returns True if [A,B] == 0 and False if [A,B] != 0.
        If two Pauli strings don't commute, they must anticommute.

        Parameters
        ----------
        PStr_A : PauliStr object
            The left (first) Pauli string in the commutator
        PStr_B : PauliStr object
            The right (second) Pauli string in the commutator

        Returns
        -------
        bool
            True if [A,B] == 0, False otherwise.

        """
        
        assert(PStr_A.N == PStr_B.N)
        
        sign1 = bool(np.sum(np.logical_and(PStr_A.X_arr, PStr_B.Z_arr)) % 2)
        sign2 = bool(np.sum(np.logical_and(PStr_A.Z_arr, PStr_B.X_arr)) % 2)
        
        if sign1 == sign2:
            return True
        else:
            return False
    
    @staticmethod
    def check_anticomm(PStr_A, PStr_B):
        """
        Checks if the two Pauli strings ANTIcommute. Returns True if {A,B} == 0 and False if {A,B} != 0.
        The ordering of A and B is unimportant, {A,B} = AB + BA.
        If two Pauli strings don't anticommute, they must anticommute.

        Parameters
        ----------
        PStr_A : PauliStr object
            One of the Pauli strings in the anticommutator
        PStr_B : PauliStr object
            The other Pauli string in the anticommutator

        Returns
        -------
        bool
            True if {A,B} == 0, False otherwise.

        """
        assert(PStr_A.N == PStr_B.N)
        
        sign1 = bool(np.sum(np.logical_and(PStr_A.X_arr, PStr_B.Z_arr)) % 2)
        sign2 = bool(np.sum(np.logical_and(PStr_A.Z_arr, PStr_B.X_arr)) % 2)
        
        if sign1 == sign2:
            return False
        else:
            return True
    
    
    @staticmethod
    def commutator(PStr_A, PStr_B):
        """
        The commutator [A,B] = AB - BA. 
        If A and B commute (AB == BA), returns the null Pauli string (the zero operator).
        Otherwise, a nonzero commutator is returned as a Pauli string.
        The commutator of any two Pauli strings is either zero or another Pauli string.

        Parameters
        ----------
        PStr_A : PauliStr object
            One of the Pauli strings in the anticommutator
        PStr_B : PauliStr object
            The other Pauli string in the anticommutator

        Returns
        -------
        PauliStr object
            The anticommutator {A,B} as a Pauli string object.

        """
        assert(PStr_A.N == PStr_B.N)
        
        if not PauliStr.check_comm(PStr_A, PStr_B):
            C = PauliStr.product(PStr_A, PStr_B)
            C.rescale(2.0)
            return C
        else:
            return PauliStr.null_str(PStr_A.N)
        
        
    @staticmethod
    def anticommutator(PStr_A, PStr_B):
        """
        The ANTIcommutator {A,B} = AB + BA. The order of A and B does not matter.
        If A and B anticommute (AB == - BA), returns the null Pauli string (the zero operator).
        Otherwise, a nonzero anticommutator is returned as a Pauli string.
        The anticommutator of any two Pauli strings is either zero or another Pauli string.

        Parameters
        ----------
        PStr_A : PauliStr object
            The left (first) Pauli string in the commutator
        PStr_B : PauliStr object
            The right (second) Pauli string in the commutator

        Returns
        -------
        PauliStr object
            The commutator [A,B] as a Pauli string object.

        """
        assert(PStr_A.N == PStr_B.N)
        
        if not PauliStr.check_anticomm(PStr_A, PStr_B):
            C = PauliStr.product(PStr_A, PStr_B)
            C.rescale(2.0)
            return C
        else:
            return PauliStr.null_str(PStr_A.N)

    



#### OLD and deprecated PauliStr methods



    def old_print_string(self):
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
                    prefactor *= -1j
                else:
                    raise ValueError("Unexpected value in Z_arr: {}, must be either 0 or 1".format(zs))
            else:
                raise ValueError("Unexpected value in X_arr: {}, must be either 0 or 1".format(xs))

        print("Pauli string: ", self.coeff*prefactor, out)
    
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
    







####### Pauli String Object
##############################################################################
##############################################################################
class Operator():
    
    def __init__(self, N=10):
        self.N = N
        self.Strings = []
    
    ### current idea is simply store a list of Pauli strings, always check if a Pauli string is already in the list...order the list? idk.

    ### include from dense / sparse method






