#define ONESITE(x, y) 2*x + y
#define TWOSITE(x, y, z, w) 8*x + 4*y + 2*z + w
#define TRACK_SIGNS 0

void print_2d (vector<vector<bool>> state) {
    // Print to console the state spectified by the vector

    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            cout << state[x][y] << " ";    
        }
        cout << endl;
    }
}

class StabilizerSet {

    void generate_Z_gates(void);
    void generate_XX_gates(void);
    void update_generators(vector<int> anticommuting);
    void update_generators_signs(vector<int> anticommuting);
    //void update_destabilizer(int row, vector<int> anticommuting);    
    //void update_destabilizer_signs(int row, vector<int> anticommuting);


    vector<vector<bool>> GF2elim(vector<vector<bool>> M);
    int pauliprod(bool x1, bool z1, bool x2, bool z2);
    int newsign(int h, int j);
    int newsign(vector<bool> X, vector<bool> Z, bool c, int row);

public:

    StabilizerSet();

    void set_string(int index, string chars, int left_pad, bool c);
    void print_generators(void);

    void apply_Z_gate(int site, int set_gate);
    void apply_XX_gate(int site1, int site2, int set_gate);

    void apply_Z_layer(void);

    void measure_Z(int site);
    void measure_XX(int site1, int site2);

    int halfchain_entanglement(void);
    int magnetization_X(void);
    int unsigned_magnetization_X(void);
    int spinglass_orderparam (void);

    vector<vector<bool>> X_arr;
    vector<vector<bool>> Z_arr;
    vector<bool> coeff;

    vector<vector<PauliStr>> XX_moves;
    vector<vector<PauliStr>> Z_moves;
};

StabilizerSet::StabilizerSet(void) : X_arr(2*L, vector<bool>(L)),
                                     Z_arr(2*L, vector<bool>(L)),
                                     coeff(2*L),
                                     Z_moves(4,  vector<PauliStr>(4,  PauliStr(1))),
                                     XX_moves(4, vector<PauliStr>(16, PauliStr(1))) {

    generate_Z_gates();
    generate_XX_gates();

}


int StabilizerSet::pauliprod(bool x1, bool z1, bool x2, bool z2) {
    /* Power to which "i" is raised when multiplying Paulis */
    if (x1 == 0 and z1 == 0) return 0;
    else if (x1 == 1 and z1 == 1) return z2 - x2;
    else if (x1 == 1 and z1 == 0) return z2*(2*x2-1);
    else return x2*(1-2*z2);
}


int StabilizerSet::newsign(int h, int j) {
    /* Find new sign when multiplying rows:
     *      R_h R_j --> R_h
     */

    int m = 0;

    for (int k = 0; k < L; k++) {
        m += pauliprod(X_arr[j][k], Z_arr[j][k], X_arr[h][k], Z_arr[h][k]);
    }

    m += 2*coeff[h] + 2*coeff[j];
    m %= 4;

    return m/2; // new sign of row h
}


int StabilizerSet::newsign(vector<bool> X, vector<bool> Z, bool c, int j) {
    /* Find new sign when multiplying rows:
     *      R_h R_j --> R_h
     */

    int m = 0;

    for (int k = 0; k < L; k++) {
        m += pauliprod(X_arr[j][k], Z_arr[j][k], X[k], Z[k]);
    }

    m += 2*c + 2*coeff[j];
    m %= 4;

    return m/2; // new sign of row h
}

void StabilizerSet::set_string(int index, string chars, int left_pad=0, bool c=0) {

    assert(left_pad+chars.length() <= L);

    for (int i = left_pad; i < (left_pad+chars.length()); i++) {

        int ci = i - left_pad;

        if ((chars[ci] == 'x') or (chars[ci] == 'X') or (chars[ci] == '1')) {
            X_arr[index][i] = true;
            Z_arr[index][i] = false;
        }
        else if ((chars[ci] == 'y') or (chars[ci] == 'Y') or (chars[ci] == '2')) {
            X_arr[index][i] = true;
            Z_arr[index][i] = true;
        }
        else if ((chars[ci] == 'z') or (chars[ci] == 'Z') or (chars[ci] == '3')) {
            X_arr[index][i] = false;
            Z_arr[index][i] = true;
        }
        else {
            X_arr[index][i] = false;
            Z_arr[index][i] = false;
        }
    }

    coeff[index] = c; 
}


void StabilizerSet::update_generators_signs(vector<int> anti) {

    int n_first = anti[0];

    for (int anti_index = 1; anti_index < anti.size(); anti_index++) {

        int n_curr  = anti[anti_index];

        // based on present configuration of X's and Z's
        coeff[n_curr] = newsign(n_first, n_curr);
    }
}


void StabilizerSet::update_generators(vector<int> anti) {

    int n_first = anti[0];

    for (int anti_index = 1; anti_index < anti.size(); anti_index++) {

        int n_curr  = anti[anti_index];

        for (int i = 0; i < L; i++) {
            X_arr[n_curr][i] = X_arr[n_curr][i] ^ X_arr[n_first][i];
            Z_arr[n_curr][i] = Z_arr[n_curr][i] ^ Z_arr[n_first][i];
        }
    }
}


vector<vector<bool>> StabilizerSet::GF2elim(vector<vector<bool>> M) {

    int i = 0;
    int j = 0;

    while ((i < L) and (j < L)) {
        // find index of max element in rest of column j
        int max = -1;
        int argmax = 0;

        for (int row = i; row < L; row++) {
            if (M[row][j] > max) {
                argmax = row;
                max = M[row][j];
            }
        }

        int k = argmax;

        // swap rows
        vector<bool> temp(M[k]);
        M[k] = M[i];
        M[i] = temp;

        vector<vector<bool>> flip(L, vector<bool>(L-j));

        for (int row = 0; row < L; row++) {
            for (int col_index = j; col_index < L; col_index++) {
                flip[row][col_index-j] = M[row][j] * M[i][col_index];
            }
        }

        for (int row = 0; row < L; row++) {
            if (row != i) {  //do not xor pivot row with itself
                for (int col_index = j; col_index < L; col_index++) {
                    M[row][col_index] = M[row][col_index] ^ flip[row][col_index-j];
                }
            }
        }

        i++;
        j++;
    }

    return M;
}


int StabilizerSet::halfchain_entanglement(void) {

    vector<vector<bool>> full_bitarr(L, vector<bool>(L));

    for (int n = 0; n < L; n++) {
        for (int i = 0; i < L/2; i++) {
            full_bitarr[n][2*i]   = X_arr[n][i];
            full_bitarr[n][2*i+1] = Z_arr[n][i];
        }
    }

    full_bitarr = GF2elim(full_bitarr);

    vector<bool> countzero(L);

    for (int i = 0; i < L; i++) {

        bool allzero = true;

        for (int j = 0; j < L; j++) {
            if (full_bitarr[i][j]) {
                allzero = false;
            }
        }

        if (allzero) {
            countzero[i] = true;
        }
    }

    int SvN = L - L/2;

    for (auto const& isAllZero : countzero) {
        SvN -= (int)isAllZero;
    }

    //assert(SvN >= 0);
    return SvN;
}


int StabilizerSet::magnetization_X (void) {
    
    int mag = 0;

    for (int i = 0; i < L; i++) {

        bool allzero = true;

        for (int n = 0; n < L; n++) {
            if (Z_arr[n][i]) allzero = false;
        }

        //cout << "allzero? " << allzero << endl;

        if (allzero) {
            
            vector<int> anti_des;
            for (int n = L; n < 2*L; n++) {
                if (Z_arr[n][i]) anti_des.push_back(n);
            }

            vector<bool> X_scratch(L);
            vector<bool> Z_scratch(L);
            bool coeff_scratch = 0;

            for (auto const& row : anti_des) {
                coeff_scratch = newsign(X_scratch, Z_scratch, coeff_scratch, row);

                for (int k = 0; k < L; k++) {
                    X_scratch[k] = X_scratch[k] ^ X_arr.at(row-L)[k];
                    Z_scratch[k] = Z_scratch[k] ^ Z_arr.at(row-L)[k];
                }
            }

            //cout << "coeff scratch " << coeff_scratch << endl;
            mag += 1-2*((int)coeff_scratch);
        }
    }

    return mag;

}


int StabilizerSet::unsigned_magnetization_X (void) {
    
    int mag = 0;

    for (int i = 0; i < L; i++) {

        bool allzero = true;

        for (int n = 0; n < L; n++) {
            if (Z_arr[n][i]) allzero = false;
        }

        if (allzero) {
            mag += 1;
        }
    }

    return mag;

}


int StabilizerSet::spinglass_orderparam (void) {
    
    int twosite = 0;
    vector<bool> onesite(L, false);

    for (int i = 0; i < L; i++) {

        bool allzero = true;

        for (int n = 0; n < L; n++) {
            if (Z_arr[n][i]) allzero = false;
        }

        if (allzero) {
            onesite[i] = true;
        }
    }

    int orderparam = 0;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {

            bool allzero = true;

            for (int n = 0; n < L; n++) {
                if (Z_arr[n][i] ^ Z_arr[n][j]) allzero = false;
            }

            orderparam -= (int)(onesite[i]*onesite[j]);

            if (allzero) {
                orderparam += 1;
            }
        }
    }

    return orderparam;

}


void StabilizerSet::measure_Z(int site) {

    // cout # stabilizers + #destabilizers that anticommute
    // look for existence of X - store location of trues 
    vector<int> anti;
    for (int n = 0; n < (1+TRACK_SIGNS)*L; n++) {
        if (X_arr[n][site]) anti.push_back(n);
    }

    // count # stabilizers that anticommute
    int anti_stab = 0;
    for (auto const& anti_pos : anti) {
        if (anti_pos < L) anti_stab += 1;
    }


    if (anti_stab > 0) {


        if (anti.size() > 1) {
            if (TRACK_SIGNS) update_generators_signs(anti);
            update_generators(anti);
        }

        // set corresponding destabilizer to first anticommuting stabilizer
        for (int i = 0; i < L; i++) {
            X_arr.at(anti[0]+L)[i] = X_arr[anti[0]][i];
            Z_arr.at(anti[0]+L)[i] = Z_arr[anti[0]][i];
        }

        coeff.at(anti[0]+L) = coeff[anti[0]];

        // set first anticommuting stabilizer to plus/minus z (random sign)
        for (int i = 0; i < L; i++) {
            X_arr[anti[0]][i] = false;
            Z_arr[anti[0]][i] = false;
        }

        Z_arr[anti[0]][site] = true;

        if (dis(gen) < 0.5) {
            coeff[anti[0]] = 0;
        }
        else {
            coeff[anti[0]] = 1;
        }

    }

}


void StabilizerSet::measure_XX(int site1, int site2) {

    // look for existence of X - store location of trues 
    vector<int> anti;

    for (int n = 0; n < (1+TRACK_SIGNS)*L; n++) {
        if (Z_arr[n][site1] ^ Z_arr[n][site2]) anti.push_back(n);
    }

    // count # stabilizers that anticommute
    int anti_stab = 0;
    for (auto const& anti_pos : anti) {
        if (anti_pos < L) anti_stab += 1;
    }

    if (anti_stab > 0) {



        if (anti.size() > 1) {
            if (TRACK_SIGNS) update_generators_signs(anti);
            update_generators(anti);
        }

        // set corresponding destabilizer to first anticommuting stabilizer
        for (int i = 0; i < L; i++) {
            X_arr.at(anti[0]+L)[i] = X_arr[anti[0]][i];
            Z_arr.at(anti[0]+L)[i] = Z_arr[anti[0]][i];
        }

        coeff.at(anti[0]+L) = coeff[anti[0]];


        // set first anticommuting stabilizer to XX
        for (int i = 0; i < L; i++) {
            X_arr[anti[0]][i] = false;
            Z_arr[anti[0]][i] = false;
        }

        X_arr[anti[0]][site1] = true;
        X_arr[anti[0]][site2] = true;

        if (dis(gen) < 0.5) {
            coeff[anti[0]] = 0;
        }
        else {
            coeff[anti[0]] = 1;
        }
    }
}


void StabilizerSet::apply_Z_gate(int site, int set_gate=-1) {
    
    int gate = (set_gate < 0) ? (int)(dis(gen)*4) : set_gate;

    for (int n = 0; n < (1+TRACK_SIGNS)*L; n++) {
        bool x = X_arr[n][site];
        bool z = Z_arr[n][site];

        auto PStr(Z_moves[gate][ONESITE(x, z)]);

        X_arr[n][site] = PStr.X_arr[0];
        Z_arr[n][site] = PStr.Z_arr[0];

        coeff[n] = coeff[n] ^ PStr.coeff;

    }
}


void StabilizerSet::apply_Z_layer(void) {
    
    vector<int> gates(L);
    generate(gates.begin(), gates.end(), []() { return gate_dis(gen); });

    for (int i = 0; i < L; i++) {
        for (int n = 0; n < (1+TRACK_SIGNS)*L; n++) {

            auto PStr(Z_moves[gates[i]][ONESITE(X_arr[n][i], Z_arr[n][i])]);

            X_arr[n][i] = PStr.X_arr[0];
            Z_arr[n][i] = PStr.Z_arr[0];

            coeff[n] = coeff[n] ^ PStr.coeff;

        }
    }
}


void StabilizerSet::apply_XX_gate(int site1, int site2, int set_gate=-1) {

    int gate = (set_gate < 0) ? (int)(dis(gen)*4) : set_gate;

    for (int n = 0; n < (1+TRACK_SIGNS)*L; n++) {
        bool x1 = X_arr[n][site1];
        bool z1 = Z_arr[n][site1];
        bool x2 = X_arr[n][site2];
        bool z2 = Z_arr[n][site2];

        X_arr[n][site1] = XX_moves[gate][TWOSITE(x1, z1, x2, z2)].X_arr[0];
        Z_arr[n][site1] = XX_moves[gate][TWOSITE(x1, z1, x2, z2)].Z_arr[0];
        X_arr[n][site2] = XX_moves[gate][TWOSITE(x1, z1, x2, z2)].X_arr[1];
        Z_arr[n][site2] = XX_moves[gate][TWOSITE(x1, z1, x2, z2)].Z_arr[1];

        coeff[n] = coeff[n] ^ XX_moves[gate][TWOSITE(x1, z1, x2, z2)].coeff;

    }
}


void StabilizerSet::generate_Z_gates(void) {
    
    PauliStr Z(1);
    Z.set_from_string("z", 0);

    vector<bool> boolarr = {0, 1};

    for (auto const& z_i : boolarr) {
        for (auto const& x_i : boolarr) {


            PauliStr PStr(1);
            PStr.set_from_XZ_string(patch::to_string(x_i)+patch::to_string(z_i), 0);
            bool commute = check_comm(Z, PStr);

            if (commute) {
                for (int g = 0; g < 4; g++) {
                    Z_moves.at(g).at(ONESITE(x_i, z_i)) = PStr;
                }
            }
            else{
                Z_moves.at(0).at(ONESITE(x_i, z_i)) = PStr;
                PStr.flip_sign();
                Z_moves.at(1).at(ONESITE(x_i, z_i)) = PStr;
                PStr.apply(Z);
                Z_moves.at(2).at(ONESITE(x_i, z_i)) = PStr;
                PStr.flip_sign();
                Z_moves.at(3).at(ONESITE(x_i, z_i)) = PStr;
            }
        }
    }
}


void StabilizerSet::generate_XX_gates(void) {
    
    PauliStr XX(2);
    XX.set_from_string("xx", 0);

    vector<bool> boolarr = {0, 1};

    for (auto const& z_j : boolarr) {
        for (auto const& x_j : boolarr) {
            for (auto const& z_i : boolarr) {
                for (auto const& x_i : boolarr) {
                    PauliStr PStr(2);
                    PStr.set_from_XZ_string(patch::to_string(x_i)+patch::to_string(z_i)+
                                            patch::to_string(x_j)+patch::to_string(z_j), 0);
                    bool commute = check_comm(XX, PStr);

                    if (commute) {
                        for (int g = 0; g < 4; g++) {
                            XX_moves.at(g).at(TWOSITE(x_i, z_i, x_j, z_j)) = PStr;
                        }
                    }
                    else{
                        XX_moves.at(0).at(TWOSITE(x_i, z_i, x_j, z_j)) = PStr;
                        PStr.flip_sign();
                        XX_moves.at(1).at(TWOSITE(x_i, z_i, x_j, z_j)) = PStr;
                        PStr.apply(XX);
                        XX_moves.at(2).at(TWOSITE(x_i, z_i, x_j, z_j)) = PStr;
                        PStr.flip_sign();
                        XX_moves.at(3).at(TWOSITE(x_i, z_i, x_j, z_j)) = PStr;
                    }
                }
            }
        }
    }
}


void StabilizerSet::print_generators(void) {

    for (int n = 0; n < 2*L; n++) {
        cout << coeff[n] << "   ";

        for (int i = 0; i < L; i++) {
            if ((X_arr[n][i] == false) and (Z_arr[n][i] == false)) {
                cout << "e";
            }
            else if ((X_arr[n][i] == true) and (Z_arr[n][i] == false)) {
                cout << "x";
            }
            else if ((X_arr[n][i] == false) and (Z_arr[n][i] == true)) {
                cout << "z";
            }
            else {
                cout << "y";
            }
        }
        cout << endl;
    }
}