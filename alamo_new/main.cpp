//
//  main.cpp
//  Alamo
//
//  Created by Maurizio Favaron on 27/11/2018.
//  Copyright © 2018 Mauri Favaron. All rights reserved.
//

#include <iostream>

#include "config.hpp"
#include "particle.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
    // Check command line arguments
    if(argc != 1) {
        cout << "Alamo - Alamo: Air LAgrangian particle dispersion Model." << endl << endl;
        cout << "Usage:" << endl << endl;
        cout << "    ./Alamo <ConfigurationFileName>" << endl << endl;
        cout << "Copyright 2018 by Servizi Territorio srl" << endl;
        cout << "This is open-source software, covered by the MIT license" << endl << endl;
        return 1;
    }
    
    // Read configuration file
    
    // Leave, no error
    return 0;
    
}
