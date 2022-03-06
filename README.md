# QubitOps
Manipulations using Pauli strings

Contents:

1. Module: Finite size qubit ops
(a) define the Pauli string basis operators for fixed number of qubits
(b) simple relations for Pauli string basis operators (rescale, multiply two strings, commute, and anticommute)
(c) define Pauli operators, which are weighted superpositions of Pauli string operators
(d) simple relations for Pauli string basis operators (addition, dot product, commutator, anticommutator, rescaling, etc)

2. Module: Inifnite size qubit ops
(a) define Pauli strings that are same as the finite size version, except now we label "site 1" of the operator
(b) include operator compression to remove leading and trailing identities, but make it optional in general
(c) define the same simple relations; however, multiplication will require extending the two Pauli strings to be the same length by padding with identities
(d) define Pauli operators and simple relations as in the finite size case


3. Module: Evolution, etc.
(a) define common unitary operator updates, clifford gates, other things that seem useful?
(b) other useful operations like partial trace, entanglement
(c) truncated versions of e^A S e^{-A} as a BCH nested commutator series?
(d) useful things for clifford circuits; generating symmetry-restricted clifford circuits?
(e) more as needed

4. Module: Wegner flow, symbolic manipulations, some of this is already written.

Future: Higher dimensions, entanglement, OTOCs, idk.
