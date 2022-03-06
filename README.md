# QubitOps
Manipulations using Pauli strings

Contents:

1. Module: "NQubitOps.py". Description: define operators acting on a system with a finite and fixed number of qubits (N). Operators are spanned by Pauli strings, which form the basis for all qubit operators. Module should include:
   - define the Pauli string basis operators for fixed number of qubits
   - simple relations for Pauli string basis operators (rescale, multiply two strings, commute, and anticommute)
   - define Pauli operators, which are weighted superpositions of Pauli string operators
   - simple relations for Pauli string basis operators (addition, dot product, commutator, anticommutator, rescaling, etc)
   - methods for time evolution, which essentially update an operator by conjugating it with particular unitaries. For Pauli strings, we can also define sets of Clifford updates including symmetry-resolved Clifford gates and how to delimit/enumerate them.
   - other operations like measurement, computing entanglement, partial traces, etc. Truncated versions of e^A S e^{-A} as a BCH nested commutator series?
   - more as needed

2. Module: "OpenQubitOps.py" or "InfQubitOps.py" or simiular. Description: similar to the finite-size module, "NQubitOps.py", except that operators and Pauli strings are defined in an open chain. The operators act on L sites, and we must also store the first site upon which the operator acts. Manipulations in an open system require additional steps like ensuring two Pauli strings have the same support before multiplying. Compressing Pauli strings by removing leading and trailing identities is desirable, but should be optional to avoid repeated extension/compression. Module should include:
   - define Pauli strings that are same as the finite size version, except now we label "site 1" of the operator
   - include operator compression to remove leading and trailing identities, but make it optional in general
   - define the same simple relations; however, multiplication will require extending the two Pauli strings to be the same length by padding with identities
   - define Pauli operators and simple relations as in the finite size case
   - Methods for time evolution and measurement, as above. They only need to act within the lightcone of the operator.

3. Module: Wegner flow, symbolic manipulations, some of this is already written.

Future: Higher dimensions, entanglement, OTOCs, idk.
