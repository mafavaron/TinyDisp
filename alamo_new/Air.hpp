//
//  Air.hpp
//  Alamo
//
//  Created by Maurizio Favaron on 28/11/2018.
//  Copyright © 2018 Mauri Favaron. All rights reserved.
//

#ifndef Air_hpp
#define Air_hpp

#include <fstream>
#include <string>
#include <vector>

using namespace std;

class Air {
private:
    static int readCSV(istream &input, vector<vector<string>> &dataSet);
};

#endif /* Air_hpp */
