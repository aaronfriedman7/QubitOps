
class PauliStr {

public:

    PauliStr(int length);

    void set_from_string(string chars, bool sign);
    void set_from_XZ_string(string chars, bool sign);
    void apply(PauliStr P);
    void dot(PauliStr P);
    void print_string(void);
    void flip_sign(void) {coeff = !coeff;};

    vector<bool> X_arr;
    vector<bool> Z_arr;
    bool coeff;
    int N;
};


PauliStr::PauliStr(int length) : X_arr(length, false),
                                 Z_arr(length, false) {

    coeff = false;
    N = length;
}


void PauliStr::set_from_string(string chars, bool sign) {

    assert(chars.length() == N);

    for (int i = 0; i < N; i++) {
        if ((chars[i] == 'x') or (chars[i] == 'X') or (chars[i] == '1')) {
            X_arr[i] = true;
            Z_arr[i] = false;
        }
        else if ((chars[i] == 'y') or (chars[i] == 'Y') or (chars[i] == '2')) {
            X_arr[i] = true;
            Z_arr[i] = true;
        }
        else if ((chars[i] == 'z') or (chars[i] == 'Z') or (chars[i] == '3')) {
            X_arr[i] = false;
            Z_arr[i] = true;
        }
        else {
            X_arr[i] = false;
            Z_arr[i] = false;
        }
    }

    coeff = sign;
}


void PauliStr::set_from_XZ_string(string chars, bool sign) {

    assert(chars.length() == 2*N);

    for (int i = 0; i < N; i++) {
        if (chars[2*i] == '0') {
            X_arr[i] = false;
        }
        else {
            X_arr[i] = true;
        }

        if (chars[2*i+1] == '0') {
            Z_arr[i] = false;
        }
        else {
            Z_arr[i] = true;
        }
    }

    coeff = sign;
}


static bool check_comm(PauliStr P1, PauliStr P2) {

    assert(P1.N == P2.N);

    bool sign;
    int accumulate = 0;

    bool sign_reverse;
    int accumulate_reverse = 0;

    for (int i = 0; i < P1.N; i++) {
        sign = P1.X_arr[i] && P2.Z_arr[i];
        accumulate += int(sign);

        sign_reverse = P1.Z_arr[i] && P2.X_arr[i];
        accumulate_reverse += int(sign_reverse);
    }

    accumulate %= 2;
    accumulate_reverse %= 2;

    return (accumulate == accumulate_reverse);

}


void PauliStr::dot(PauliStr P) {
    /* Apply Pauli string object to the right of the present instance.
     * 
     *      PStr.apply(P) = PStr.P
     * 
     * The string of X's and Z's is updated according to
     * 
     *      X_new = X1 xor X2
     *      Z_new = Z1 xor Z2
     *
     * The sign bit is updated by commuting X's though Z's and multiplying
     * the implicit imaginary units that are present for XZ = -1j Y
     */ 

    bool sign, complex;
    int flip = 0;

    for (int i = 0; i < N; i++) {
        sign = Z_arr[i] && P.X_arr[i];
        complex = X_arr[i] && Z_arr[i] && P.X_arr[i] && P.Z_arr[i];
        flip += int(sign) + int(complex);
    }

    for (int i = 0; i < N; i++) {
        X_arr[i] = X_arr[i] ^ P.X_arr[i];
        Z_arr[i] = Z_arr[i] ^ P.Z_arr[i];
    }


    flip %= 2;

    if (flip) {
        coeff = (!coeff) ^ P.coeff;
    }
    else {
        coeff = coeff ^ P.coeff;
    }

}


void PauliStr::apply(PauliStr P) {
    /* Apply Pauli string object to the left of the present instance.
     *
     *      PStr.apply(P) = P.PStr
     *
     * The string of X's and Z's is updated according to
     * 
     *      X_new = X1 xor X2
     *      Z_new = Z1 xor Z2
     *
     * The sign bit is updated by commuting X's though Z's and multiplying
     * the implicit imaginary units that are present for XZ = -1j Y
     *
     */

    bool sign, complex;
    int flip = 0;

    for (int i = 0; i < N; i++) {
        sign = X_arr[i] && P.Z_arr[i];
        complex = X_arr[i] && Z_arr[i] && P.X_arr[i] && P.Z_arr[i];
        flip += int(sign) + int(complex);
    }

    for (int i = 0; i < N; i++) {
        X_arr[i] = X_arr[i] ^ P.X_arr[i];
        Z_arr[i] = Z_arr[i] ^ P.Z_arr[i];
    }


    flip %= 2;

    if (flip) {
        coeff = (!coeff) ^ P.coeff;
    }
    else {
        coeff = coeff ^ P.coeff;
    }

}


void PauliStr::print_string(void) {

    cout << "Sign: " << coeff << "   ";

    for (int i = 0; i < N; i++) {
        if (X_arr[i] == false and Z_arr[i] == false) {
            cout << 'e';
        }
        else if (X_arr[i] == true and Z_arr[i] == false) {
            cout << 'x';
        }
        else if (X_arr[i] == false and Z_arr[i] == true) {
            cout << 'z';
        }
        else {
            cout << 'y';
        }
    }

    cout << endl;
}
