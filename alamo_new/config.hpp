//
//  config.hpp
//  Alamo
//
//  Created by Maurizio Favaron on 27/11/2018.
//  Copyright © 2018 Mauri Favaron. All rights reserved.
//

#ifndef config_hpp
#define config_hpp

#include <string>

using namespace std;

class Config {
private:
    bool   isComplete;  // True if configuration is complete, false otherwise
    int    iErrCode;    // On exit, it contains an error code (0 in case of success)
    double x0, y0;      // Coordinates of receptor grid SW point (m)
    long   nx, ny;      // Number of receptor grid elements along X and Y axes
    double dx, dy;      // Grid spacings along X and Y axes (m)
    double zMax;        // Maximum height (m)
    long Nstep;         // Number of integration sub-steps within a single averaging step
    double Tmed;        // Averaging step (s)
    long Np;            // Number of particles released per sub-step
    string sourcesFile; // Name of source emission data file
    string emissFile;   // Name of source emission profiles data file (may be empty for steady emissions)
    string meteoFile;   // Name of file containing surface meteorological data
    string profileFile; // Name of file containing upper-air (SODAR) meteorological data
    double zlev;        // Station height above mean sea level (m)
    double z0;          // Aerodynamic roughness length
    double zr;          // Anemometer height above ground (m)
    string Fileout;     // Name of concentration file
    string Filedia;     // Name of diagnostic file
    double fat;         // Concentration scaling factor
public:
    Config(const string &iniFileName);
    virtual ~Config();
};

#endif /* config_hpp */
