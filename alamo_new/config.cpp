//
//  config.cpp
//  Alamo
//
//  Created by Maurizio Favaron on 27/11/2018.
//  Copyright © 2018 Mauri Favaron. All rights reserved.
//

#include "config.hpp"

#include "iniReader.h"

Config::Config(const string &iniFileName) {
    
    // Assume success(will falsify on failure)
    this->iErrCode = 0;
    this->isComplete = false;
    
    // Try gathering configuration
    INIReader reader(iniFileName);
    if(reader.ParseError()) {
        this->iErrCode = 1;
        return;
    }
    // -1- Grid
    this->x0 = reader.GetReal(string("Grid"), string("X0"), -9999.9);
    this->y0 = reader.GetReal(string("Grid"), string("Y0"), -9999.9);
    if(this->x0 < -9000.0 | this->y0 < -9000.0) {
        this->iErrCode = 2;
        return;
    }
    this->nx = reader.GetInteger(string("Grid"), string("nx"), -9999);
    this->ny = reader.GetInteger(string("Grid"), string("ny"), -9999);
    if(this->nx < 0 | this->ny < 0) {
        this->iErrCode = 3;
        return;
    }
    this->dx = reader.GetReal(string("Grid"), string("dx"), -9999.9);
    this->dy = reader.GetReal(string("Grid"), string("dy"), -9999.9);
    if(this->dx < 0.0 | this->dy < 0.0) {
        this->iErrCode = 4;
        return;
    }
    this->zMax = reader.GetReal(string("Grid"), string("zMax"), -9999.9);
    if(this->zMax <= 0.0) {
        this->iErrCode = 5;
        return;
    }
    // -1- Timing and rates
    this->Tmed = reader.GetReal(string("Timing"), string("AveragingTime"), -9999.9);
    if(this->Tmed <= 0.0) {
        this->iErrCode = 6;
        return;
    }
    this->Nstep = reader.GetInteger(string("Timing"), string("SubstepsInAvgStep"), -9999);
    if(this->Nstep <= 0) {
        this->iErrCode = 7;
        return;
    }
    this->Np = reader.GetInteger(string("Timing"), string("NumParticlesPerSourcePerSubstep"), -9999);
    if(this->Np <= 0) {
        this->iErrCode = 8;
        return;
    }

};


Config::~Config() {};

