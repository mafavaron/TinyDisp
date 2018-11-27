//
//  td - TinyDisp Lagrangian particle atmospheric dispersion model
//
//  Created by Mauri Favaron on 09/04/18.
//  Copyright © 2018 Mauri Favaron.
//  This is open source software.
//
// TinyDisp is an evolution of ALAMO model, written by prof. Roberto Sozzi.
// My contribution is, to clarify the code and its internal working in view
// of explaining what a conventional Lagrangian particle model is in its very essence.
//
// To a very large extent, the simple architecture of orignal ALAMO code
// has been retained.
//

#include <iostream>

using namespace std;

int main(int argc, const char * argv[]) {
    
	// Check command line arguments
	if(argc != 1) {
		cout << "td - TinyDisp: a lightweight Lagrangian particle atmospheric dispersion model." << endl << endl;
		cout << "Usage:" << endl << endl;
		cout << "    ./td <ConfigurationFileName>" << endl << endl;
		cout << "Copyright 2018 by Mauri Favaron" << endl;
		cout << "This is open-source software, covered by the MIT license" << endl << endl;
        return 1;
    }
    
    // Read configuration file
    
    // Leave, no error
    return 0;
}
