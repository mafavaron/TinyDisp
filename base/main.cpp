//
//  td - TinyDisp Lagrangian particle atmospheric dispersion model
//
//  Created by Mauri Favaron on 09/04/18.
//  Copyright © 2018 Mauri Favaron.
//  This is open source software.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

int main(int argc, const char * argv[]) {
    
    // Check command line arguments
    if(argc != 1) {
        printf("td - TinyDisp: a lightweight Lagrangian particle atmospheric dispersion model.\n");
        printf("\n");
        printf("Usage:\n");
        printf("\n");
        printf("    ./td <ConfigurationFileName>\n");
        printf("\n");
        printf("Copyright 2018 by Servizi Territorio srl\n");
        printf("This is open-source software, covered by the MIT license\n");
        printf("\n");
        return 1;
    }
    
    // Read configuration file
    
    // Leave, no error
    return 0;
}
