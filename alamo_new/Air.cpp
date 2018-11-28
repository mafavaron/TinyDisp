//
//  Air.cpp
//  Alamo
//
//  Created by Maurizio Favaron on 28/11/2018.
//  Copyright © 2018 Mauri Favaron. All rights reserved.
//

#include "Air.hpp"

#include <sstream>

int Air::readCSV(istream &input, vector<vector<string>> &dataSet) {
    
    // Read lines from input file
    string csvLine;
    while( getline(input, csvLine)) {
        istringstream csvStream(csvLine);
        vector<string> csvColumn;
        string csvElement;
        while(getline(csvStream, csvElement, ',')) {
            csvColumn.push_back(csvElement);
        }
        dataSet.push_back(csvColumn);
    }
};
