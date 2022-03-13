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


#define L 18

#include "NQubitOps.h"
#include "StabilizerSet.h"
#include "functions.h"
#include "io_functions.h"

int main () {

	int approx = (int) (50*(exp(L/3.0)));//400*L;

	cout << approx << endl;

	int depth = 18000;
	int Nhist = 1000;

	double p = 0.75;

	vector<int> sites(L);

	vector<double> Mx(depth);

	for (int i = 0; i < L; i++) {
		sites.at(i) = i;
	}

	for (int rep = 0; rep < Nhist; rep++) {

		cout << rep << endl;

		// create empty set of stabilizers
		StabilizerSet S;

		// populate with z generators
		for (int i = 0; i < L; i++) {
		    S.set_string(i, "x", i);
		}

		// evolve circuit in time
		for (int t = 0; t < depth; t++){

			Mx.at(t) += S.unsigned_magnetization_X();

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
	}

	// normalize by the number of histories
	for (int t = 0; t < depth; t++) {
		Mx[t] /= ((double) Nhist);
	}

	save_to_file("Mx_nosign_L18_p075_d18000_Nhist1000.txt", Mx);

	return 0;
}