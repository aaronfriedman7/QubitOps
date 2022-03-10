template<class A>
void save_to_file (string filename, vector<A> &vect) {
    ofstream outputfile(filename);
    outputfile << std::setprecision(6);
    for (const auto &e : vect) outputfile << e << "\n";
}

template<class A>
void save2d_to_file (string filename, vector<vector<A>> &vect) {
    ofstream outputfile(filename);

    for (auto& vt : vect) {
        /*for (auto& elem : vt) {
            outputfile << elem << ",";
        }*/
        if (vt.size() > 0) {
            for (auto iter = vt.begin(); iter != std::prev(vt.end()); ++iter){
                outputfile << *iter << " ";
            }
            outputfile << vt.back() << "\n";
        }
        else {
            outputfile << "\n";
        }
    }
}

vector<int> load_from_file (string filename) {
    ifstream infile(filename);
    istream_iterator<double> start(infile), end;
    vector<int> output(start, end);
    return output;
}

vector<vector<int>> load2d_from_file (string filename) {
    vector<vector<int>> vec;

    std::ifstream file_in(filename);
    if (!file_in) {/*error*/}

    std::string line;
    while (std::getline(file_in, line)) // Read next line to "line", stop if no more lines.
    {
        // Construct string stream from "line", see while loop below for usage.
        std::istringstream ss(line);

        vec.push_back({}); // Add one more empty vector (of vectors) to "vec".

        int x;
        while (ss >> x) // Read next int from "ss" to "x", stop if no more ints.
            vec.back().push_back(x); // Add it to the last sub-vector of "vec".
    }

    return vec;
}
