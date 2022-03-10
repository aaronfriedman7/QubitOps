#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <iterator>


using namespace std;

// Set up random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);
std::uniform_int_distribution<> gate_dis(0, 3);

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}


#define L 128

#include "NQubitOps.h"
#include "StabilizerSet.h"
#include "functions.h"
#include "io_functions.h"

int main () {

	int depth = 5*L;
	int Nmeasure = 20;
	int Nhist = 100;

	vector<int> sites(L);
	vector<double> av_entanglement(Nmeasure);
	vector<double> pvals = linspace(0.05, 1.0, Nmeasure);

	for (int i = 0; i < L; i++) {
		sites.at(i) = i;
	}

	for (int rep = 0; rep < Nhist; rep++) {

		cout << rep << endl;

		for (int p_index = 0; p_index < Nmeasure; p_index++) {

			double p = pvals.at(p_index);

			// create empty set of stabilizers
			StabilizerSet S;

			// populate with z generators
			for (int i = 0; i < L; i++) {
			    S.set_string(i, "z", i);
			}

			// evolve circuit in time
			for (int t = 0; t < depth; t++){

				// randomly shuffle the sites to generate random pairs
				random_shuffle(sites.begin(), sites.end());

				
				// apply single site Z gates to all sites
				for (int i = 0; i < L; i++) {
					S.apply_Z_gate(i);
				}

				// apply XX gates to random pairs of sites
				for (int i = 0; i < L/2; i++) {
					S.apply_XX_gate(sites[2*i], sites[2*i+1]);
				}
				
				
				// randomly measure
				for (int i = 0; i < L/2; i++) {
					if (dis(gen) < p) S.measure_XX(2*i, 2*i+1);
				}
				for (int i = 0; i < L/2-1; i++) {
					if (dis(gen) < p) S.measure_XX(2*i+1, 2*i+2);
				}
			}

			// "sample" the entanglement (in the monte carlo sense)
			av_entanglement[p_index] += S.halfchain_entanglement();
		}
	}

	// normalize by the number of histories
	for (int p_index = 0; p_index < Nmeasure; p_index++) {
		av_entanglement[p_index] /= ((double) Nhist);
	}

	save_to_file("entanglement_test_L128.txt", av_entanglement);

	return 0;
}