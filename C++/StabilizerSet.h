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

    vector<vector<bool>> GF2elim(vector<vector<bool>> M);

public:

    StabilizerSet();

    void set_string(int index, string chars, int left_pad);
    void print_generators(void);

    void apply_Z_gate(int site, int set_gate);
    void apply_XX_gate(int site1, int site2, int set_gate);

    void apply_Z_layer(void);

    void measure_Z(int site);
    void measure_XX(int site1, int site2);

    int halfchain_entanglement(void);

    vector<vector<bool>> X_arr;
    vector<vector<bool>> Z_arr;
    vector<bool> coeff;

    vector<vector<PauliStr>> XX_moves;
    vector<vector<PauliStr>> Z_moves;
};

StabilizerSet::StabilizerSet(void) : X_arr(L, vector<bool>(L)),
                                     Z_arr(L, vector<bool>(L)),
                                     coeff(L),
                                     Z_moves(4,  vector<PauliStr>(4,  PauliStr(1))),
                                     XX_moves(4, vector<PauliStr>(16, PauliStr(1))) {

    generate_Z_gates();
    generate_XX_gates();

}


void StabilizerSet::set_string(int index, string chars, int left_pad=0) {

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
}


void StabilizerSet::update_generators(vector<int> anti) {

    if (TRACK_SIGNS) {
        vector<bool> coeff_copy(coeff);

        for (int anti_index = 1; anti_index < anti.size(); anti_index++) {

            bool sign, complex;
            int flip = 0;

            int n_prev = anti[anti_index-1];
            int n_curr = anti[anti_index];

            for (int i = 0; i < L; i++) {
                sign = Z_arr[n_prev][i] && X_arr[n_curr][i];
                complex = X_arr[n_prev][i] && Z_arr[n_prev][i] &&
                          X_arr[n_curr][i] && Z_arr[n_curr][i];
                flip += int(sign) + int(complex);
            }

            flip %= 2;

            if (flip) {
                coeff[n_curr] = (!coeff_copy[n_prev]) ^ coeff_copy[n_curr];
            }
            else {
                coeff[n_curr] = (coeff_copy[n_prev]) ^ coeff_copy[n_curr];
            }

        }
    }

    auto X_copy(X_arr);
    auto Z_copy(Z_arr);

    for (int anti_index = 1; anti_index < anti.size(); anti_index++) {

        int n_prev = anti[anti_index-1];
        int n_curr = anti[anti_index];

        for (int i = 0; i < L; i++) {
            X_arr[n_curr][i] = X_copy[n_curr][i] ^ X_copy[n_prev][i];
            Z_arr[n_curr][i] = Z_copy[n_curr][i] ^ Z_copy[n_prev][i];
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

    assert(SvN > 0);
    return SvN;
}


void StabilizerSet::measure_Z(int site) {

    // look for existence of X - store location of trues 
    vector<int> anti;

    for (int n = 0; n < L; n++) {
        if (X_arr[n][site] == true) {
            anti.push_back(n);
        }
    }

    if (anti.size() > 0) {
        if (anti.size() > 1) {
            update_generators(anti);
        }

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

    for (int n = 0; n < L; n++) {
        if (Z_arr[n][site1] ^ Z_arr[n][site2]) {
            anti.push_back(n);
        }
    }

    if (anti.size() > 0) {
        if (anti.size() > 1) {
            update_generators(anti);
        }

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

    for (int n = 0; n < L; n++) {
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
        for (int n = 0; n < L; n++) {
            //bool x = X_arr[n][i];
            //bool z = Z_arr[n][i];

            auto PStr(Z_moves[gates[i]][ONESITE(X_arr[n][i], Z_arr[n][i])]);

            X_arr[n][i] = PStr.X_arr[0];
            Z_arr[n][i] = PStr.Z_arr[0];

            coeff[n] = coeff[n] ^ PStr.coeff;
        }
    }
}


void StabilizerSet::apply_XX_gate(int site1, int site2, int set_gate=-1) {

    int gate = (set_gate < 0) ? (int)(dis(gen)*4) : set_gate;

    for (int n = 0; n < L; n++) {
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

    for (int n = 0; n < L; n++) {
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