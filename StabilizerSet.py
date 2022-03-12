#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:34:49 2020

@author: Aaron + Ollie

Disclaimer: For internal / private use only; please do not share without my permission.
"""
import numpy as np
import numba
import random
import NQubitOps as NQO

@numba.jit(nopython=True)
def GF2elim(M):

    m, n = M.shape

    i = 0
    j = 0

    while i < m and j < n:
        # find index of max element in rest of column j
        k = np.argmax(M[i:, j]) + i

        # swap rows
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp

        aijn = M[i, j:]

        col = np.copy(M[:, j])
        col[i] = 0 #do not xor pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        i += 1
        j += 1

    return M


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

        self.generate_Z_gates()
        self.generate_XX_gates()


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
        self.coeff[index] = coeff


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

        assert(self.N == PStr_B.N)

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

        assert(self.N == PStr_B.N)

        sign_arr = np.logical_and(PStr_B.Z_arr[np.newaxis, :], self.X_arr)
        self.coeff = self.coeff * PStr_B.coeff * (-1)**(np.sum(sign_arr, axis=1) % 2)

        self.X_arr = np.logical_xor(PStr_B.X_arr[np.newaxis, :], self.X_arr)
        self.Z_arr = np.logical_xor(PStr_B.Z_arr[np.newaxis, :], self.Z_arr)


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

        assert(self.N == PStr_B.N)

        sign_arr = np.logical_and(self.Z_arr[index, :], PStr_B.X_arr)
        self.coeff[index] = self.coeff[index] * PStr_B.coeff * (-1)**(np.sum(sign_arr) % 2)

        self.X_arr[index, :] = np.logical_xor(self.X_arr[index, :], PStr_B.X_arr)
        self.Z_arr[index, :] = np.logical_xor(self.Z_arr[index, :], PStr_B.Z_arr)


    def entanglement_entropy(self):
        """
        Half chain entanglement entropy.
    

        Find the entanglement entropy of the state specified by the set of
        stabilizers. This works by performing a Gaussian elimination to find
        the left-endpoints in the "clipped gauge". The formula for the
        entanglement entropy is:

            S_A(x) = sum_{y <= x} (rho(y) - 1)

        where rho(y) is the density of left endpoints on site y.

        -------
        output: float
            value of the entanglement entropy (logarithm base 2)
        """


        # Put all X's and Z's into one array, with X's preceding Z's
        full_bitarr = np.zeros((self.N, 2*self.N), dtype='int')
        full_bitarr[:, ::2] = self.X_arr
        full_bitarr[:, 1::2] = self.Z_arr

        full_bitarr = GF2elim(full_bitarr[:, :self.N])

        allzero = np.all(np.logical_not(full_bitarr), axis=1)

        SvN = self.N - np.sum(allzero) - self.N//2

        return SvN


    def random_Z_gate(self, pos, gate=None):
        r"""
        Apply gate of the form U = exp(i h Z) to the state of the system.

        At the level of the stabilizers, this leads to a transformation 

            S -> U^dagger S U

        This leads to the following possiblilties:

            X -> \pm X
        or  X -> \pm XZ
        
        therefore (01) -> (01) always
        while     (10) -> (10) or (11) with a potential sign change    

        Parameters
        ----------
        pos : integer
            location at which the gate is to be applied
        gate : integer, optional
            gate to be applied from the list of possible unitaries

        """

        # random integer from {0, 1, 2, 3} (= #gates) if gate not supplied
        rand = gate if gate is not None else random.randint(0, 3) 

        new_strings = [self.Z_moves[(self.X_arr[n, pos],
                                     self.Z_arr[n, pos])][rand] for n in range(self.N)]

        signs = np.array([new_strings[n].coeff for n in range(self.N)])
        new_X = np.array([new_strings[n].X_arr[0] for n in range(self.N)])
        new_Z = np.array([new_strings[n].Z_arr[0] for n in range(self.N)])

        self.X_arr[:, pos] = new_X
        self.Z_arr[:, pos] = new_Z
        self.coeff *= signs


    def random_XX_gate(self, pos1, pos2, gate=None):
        r"""
        Apply gate of the form U = exp(i J Xi Xj) to the state of the system.

        At the level of the stabilizers, this leads to a transformation 

            S -> U^dagger S U

        This leads to the following possiblilties:

            Zi -> \pm Zi
        or  Zi -> \pm Zi Xi Xj

        Parameters
        ----------
        pos1 : integer
            first location at which two site gate is to be applied
        pos2 : integer
            second location at which two site gate is to be applied
        gate : integer, optional
            gate to be applied from list of possible unitaries

        """

        # random integer from {0, 1, 2, 3} (= #gates) if gate not supplied
        rand = gate if gate is not None else random.randint(0, 3) 

        new_strings = [self.XX_moves[(self.X_arr[n, pos1], self.Z_arr[n, pos1],
                                      self.X_arr[n, pos2], self.Z_arr[n, pos2])][rand] for n in range(self.N)]

        signs = np.array([new_strings[n].coeff for n in range(self.N)])
        new_X = np.array([new_strings[n].X_arr for n in range(self.N)])
        new_Z = np.array([new_strings[n].Z_arr for n in range(self.N)])

        self.X_arr[:, [pos1, pos2]] = new_X
        self.Z_arr[:, [pos1, pos2]] = new_Z
        self.coeff *= signs


    def measure_Z(self, pos):
        """
        Perform a measurement on a single site, specified by the index "pos" 

        Suppose that stabilizers g_i with i > k anticommute with Z, then
        new set of stabilizers is:

            {g_1, g_2, ..., g_k, g_{k+1} g_{k+2}, ..., g_{N-1} g_N, pm Z}

        Computationally, we shift the array by one position

        [ g_{k+1} ]
                   \
        [ g_{k+2} ]  [ g_{k+2} ]
                   \
                     [ g_{k+3} ] etc.

        Pauli string multiplication uses similar code to the dot() function.
        Function takes O(n^2) time

        Parameters
        ----------
        pos : integer
            location at which two site gate is to be applied

        """

        # presence of X implies anticommutativity with Z
        anti = np.where(self.X_arr[:, pos] == True)[0]

        if anti.size > 0:
            # anticommutes so projected state is different from original

            self.X_arr[anti[1:]] = np.logical_xor(self.X_arr[anti[1:], :],
                                                  self.X_arr[anti[:-1], :])
            self.Z_arr[anti[1:]] = np.logical_xor(self.Z_arr[anti[1:], :],
                                                  self.Z_arr[anti[:-1], :])

            sign_arr = np.logical_and(self.Z_arr[anti[1:], :],
                                      self.X_arr[anti[:-1], :])
            self.coeff[anti[1:]] = (self.coeff[anti[1:]] *
                                    self.coeff[anti[:-1]] *
                                    (-1)**(np.sum(sign_arr, axis=1) % 2))

            where_first = anti[0]

            self.X_arr[where_first, :] = np.zeros(self.N)
            self.Z_arr[where_first, :] = np.zeros(self.N)
            self.Z_arr[where_first, pos] = 1

            if random.randint(0, 1):
                self.coeff[where_first] = 1
            else:
                self.coeff[where_first] = -1


    def measure_XX(self, pos1, pos2):
        """
        Perform a measurement on two sites, specified by the indices "pos1"
        and "pos2" 

        Suppose that stabilizers g_i with i > k anticommute with XX, then
        new set of stabilizers is:

            {g_1, g_2, ..., g_k, g_{k+1} g_{k+2}, ..., g_{N-1} g_N, pm Z}

        Computationally, we shift the array by one position

        [ g_{k+1} ]
                   \
        [ g_{k+2} ]  [ g_{k+2} ]
                   \
                     [ g_{k+3} ] etc.

        Pauli string multiplication uses similar code to the dot() function.
        Function takes O(n^2) time.

        Parameters
        ----------
        pos1 : integer
            first location at which two site gate is to be applied
        pos2 : integer
            second location at which two site gate is to be applied

        """

        # presence of exactly one Z implies anticommutativity with XX
        anti = np.where(np.logical_xor(self.Z_arr[:, pos1],
                                       self.Z_arr[:, pos2]) == True)[0]

        if anti.size > 0:
            # anticommutes so projected state is different from original

            self.X_arr[anti[1:]] = np.logical_xor(self.X_arr[anti[1:], :],
                                                  self.X_arr[anti[:-1], :])
            self.Z_arr[anti[1:]] = np.logical_xor(self.Z_arr[anti[1:], :],
                                                  self.Z_arr[anti[:-1], :])

            sign_arr = np.logical_and(self.Z_arr[anti[1:], :],
                                      self.X_arr[anti[:-1], :])
            self.coeff[anti[1:]] = (self.coeff[anti[1:]] *
                                    self.coeff[anti[:-1]] *
                                    (-1)**(np.sum(sign_arr, axis=1) % 2))

            where_first = anti[0]

            self.X_arr[where_first, :] = np.zeros(self.N)
            self.Z_arr[where_first, :] = np.zeros(self.N)
            self.X_arr[where_first, pos1] = True
            self.X_arr[where_first, pos2] = True

            if random.randint(0, 1):
                self.coeff[where_first] = 1
            else:
                self.coeff[where_first] = -1


    def X_magnetization(self):
        """
        Magnetisation in the X direction. Not normalized, so gives

            < sum_i X_i >

        Returns
        ----------
        output : integer
            expectation value of magnetisation, equal to the number of 
            sites i satisfying all(z_i) = 0
        """

        # test whether all equal to 0 at particular position for all strings
        all_vanish = np.all(np.logical_not(self.Z_arr), axis=0)

        return np.sum(all_vanish)


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
                        raise ValueError("Unexpected value in Z_arr: {}, "+
                                         "must be either 0 or 1".format(zs))
                elif (xs == 1):
                    if (zs == 0):
                        out += "x"
                    elif (zs == 1):
                        out += "y"
                        prefactor *= (-1j)
                    else:
                        raise ValueError("Unexpected value in Z_arr: {}, "+
                                         "must be either 0 or 1".format(zs))
                else:
                    raise ValueError("Unexpected value in X_arr: {}, "+
                                     "must be either 0 or 1".format(xs))

            print(self.coeff[i]*prefactor, out)


    def generate_XX_gates(self):
        """
        Generate the allowed transitions between Pauli strings as a
        dictionary. The input (key) is a tuple of integers that correspond to
        the initial Pauli strings on two sites, i and j:

            (x_i z_i) ; (x_j z_j)

        The output of the dictionary is a list of output strings generated
        by the XX gates belonging to the Clifford group:

            (x_i z_i) ; (x_j z_j) --> U^dagger (x_i z_i) ; (x_j z_j) U

        Each list has the same length, which corresonds to the number of
        unitaries. These unitaries are

            U[k] = exp(-i J[k] X_i X_j),

        where J[k] belongs to {0, pi/2, pi, 3 pi/2}.

        Make use of the BCH identity, which gives

            U^dagger[k] S U[k] = S, if [S, Z_i] = 0
                            or = [cos(h) + 1j sin(h) Z_i] S, if {S, Z_i} = 0

        """

        XX_string = NQO.PauliStr.from_char_string('xx')

        self.XX_moves = {}

        for z_j in [0, 1]:
            for x_j in [0, 1]:
                for z_i in [0, 1]:
                    for x_i in [0, 1]:
                        
                        self.XX_moves[(x_i, z_i, x_j, z_j)] = []
                        
                        PStr = NQO.PauliStr.from_XZ_string(str(x_i)+str(z_i)+str(x_j)+str(z_j))
                        commute = NQO.PauliStr.check_comm(XX_string, PStr)

                        if commute:
                            self.XX_moves[(x_i, z_i, x_j, z_j)] += [NQO.PauliStr.copy(PStr)]*4

                        else:
                            # cos(0) (x_i z_i) ; (x_j z_j)
                            self.XX_moves[(x_i, z_i, x_j, z_j)].append(NQO.PauliStr.copy(PStr))

                            PStr.rescale(-1)
                            # cos(pi) (x_i z_i) ; (x_j z_j)
                            self.XX_moves[(x_i, z_i, x_j, z_j)].append(NQO.PauliStr.copy(PStr))

                            PStr.apply(XX_string)
                            PStr.rescale(1j)
                            # 1j sin(pi/2) (X_i X_j) (x_i z_i) ; (x_j z_j)
                            self.XX_moves[(x_i, z_i, x_j, z_j)].append(NQO.PauliStr.copy(PStr))

                            PStr.rescale(-1)
                            # 1j sin(3 pi/2) (X_i X_j) (x_i z_i) ; (x_j z_j)
                            self.XX_moves[(x_i, z_i, x_j, z_j)].append(NQO.PauliStr.copy(PStr))


    def generate_Z_gates(self):
        """
        Generate the allowed transitions between Pauli strings as a
        dictionary. The input (key) is a tuple of integers that correspond to
        the initial Pauli strings on one site, labelled i:

            (x_i z_i)

        The output of the dictionary is a list of output strings generated
        by the Z gates belonging to the Clifford group:

            (x_i z_i) --> U^dagger (x_i z_i) U

        Each list has the same length, which corresonds to the number of
        unitaries. These unitaries are

            U[k] = exp(-i h[k] Z_i),

        where h[k] belongs to {0, pi/2, pi, 3 pi/2}.

        Make use of the BCH identity, which gives

            U^dagger[k] S U[k] = S, if [S, Z_i] = 0
                            or = [cos(h) + 1j sin(h) Z_i] S, if {S, Z_i} = 0

        """

        Z_string = NQO.PauliStr.from_char_string('z')

        self.Z_moves = {}

        for z_i in [0, 1]:
            for x_i in [0, 1]:

                self.Z_moves[(x_i, z_i)] = []

                PStr = NQO.PauliStr.from_XZ_string(str(x_i)+str(z_i))
                commute = NQO.PauliStr.check_comm(Z_string, PStr)

                if commute:
                    self.Z_moves[(x_i, z_i)] += [NQO.PauliStr.copy(PStr)]*4

                else:
                    # cos(0) (x_i z_i)
                    self.Z_moves[(x_i, z_i)].append(NQO.PauliStr.copy(PStr))

                    PStr.rescale(-1)
                    # cos(pi) (x_i z_i)
                    self.Z_moves[(x_i, z_i)].append(NQO.PauliStr.copy(PStr))

                    PStr.apply(Z_string)
                    PStr.rescale(1j)
                    # 1j sin(pi/2) Z_i (x_i z_i)
                    self.Z_moves[(x_i, z_i)].append(NQO.PauliStr.copy(PStr))

                    PStr.rescale(-1)
                    # 1j sin(3 pi/2) Z_i (x_i z_i)
                    self.Z_moves[(x_i, z_i)].append(NQO.PauliStr.copy(PStr))


def stabilizer_array(stabilizer):
    Op = stabilizer.to_ndarray()
    (d1,d2) = np.shape(Op)
    assert d1==d2
    Op += np.eye(d1)
    return 0.5*Op


def dmat_array(N, stabilizers):
    assert N == len(stabilizers)
    
    rho = stabilizer_array(stabilizers[0])
    
    for foo in range(1,len(stabilizers)):
        newguy = stabilizer_array(stabilizers[foo])
        rho = np.matmul(rho,newguy)
    
    return rho
    
    
    
    
    
    
    