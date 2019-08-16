#include "cfg.h"

Cfg::Cfg() {

  this->iDebugLevel = 4;
  this->sDiaFile = "";
  this->iFrameInterval = 0;
  this->sFramePath = "";
  this->iExecMode = 0;
  this->iAvgTime = 0;
  this->iNumStep = 0;
  this->iNumPart = 0;
  this->iMaxAge = 0;
  this->sStatic = "";
  this->sDynamic = "";
  this->sMetInpFile = "";
  this->sMetOutFile = "";
  this->sMetDiaFile = "";
  this->rHeight = 0.0;
  this->rZ0 = 0.0;
  this->rZr = 0.0;
  this->rZt = 0.0;
  this->rGamma = 0.0;
  this->iHemisphere = 0;
  this->sConcFile = "";
  this->rX0 = 0.0;
  this->rY0 = 0.0;
  this->iNx = 0;
  this->iNy = 0;
  this->rDx = 0.0;
  this->rDy = 0.0;
  this->iNz = 0;
  this->rDz = 0.0;

  this->iState = -1;

}


Cfg::Cfg(std::ifstream& cfg) {

  // Assume an invalid configuration
  this->iState = 1;

  // Read configuration
  if(cfg.is_open()) {

    char* buffer;
    buffer = new char[257];
    buffer[256] = '\0';

    // General
    cfg.read((char*)&this->iDebugLevel, sizeof(int));
    cfg.read(buffer, 256);
    this->sDiaFile = buffer;
    cfg.read((char*)&this->iFrameInterval, sizeof(int));
    cfg.read(buffer, 256);
    this->sFramePath = buffer;
    cfg.read((char*)&this->iExecMode, sizeof(&(this->iExecMode)));

    // Timing
    cfg.read((char*)&this->iAvgTime, sizeof(&(this->iAvgTime)));
    cfg.read((char*)&this->iNumStep, sizeof(&(this->iNumStep)));
    cfg.read((char*)&this->iNumPart, sizeof(&(this->iNumPart)));
    cfg.read((char*)&this->iMaxAge, sizeof(&(this->iMaxAge)));

    // Emission
    cfg.read(buffer, 256);
    this->sStatic = buffer;
    cfg.read(buffer, 256);
    this->sDynamic = buffer;

    // Meteo
    cfg.read(buffer, 256);
    this->sMetInpFile = buffer;
    cfg.read(buffer, 256);
    this->sMetOutFile = buffer;
    cfg.read(buffer, 256);
    this->sMetDiaFile = buffer;
    cfg.read((char*)&this->rZ0, sizeof(&(this->rZ0)));
    cfg.read((char*)&this->rZr, sizeof(&(this->rZr)));
    cfg.read((char*)&this->rZt, sizeof(&(this->rZt)));
    cfg.read((char*)&this->rGamma, sizeof(&(this->rGamma)));
    cfg.read((char*)&this->iHemisphere, sizeof(&(this->iHemisphere)));

    // Output
    cfg.read(buffer, 256);
    this->sConcFile = buffer;
    cfg.read((char*)&this->rX0, sizeof(&(this->rX0)));
    cfg.read((char*)&this->rY0, sizeof(&(this->rY0)));
    cfg.read((char*)&this->iNx, sizeof(&(this->iNx)));
    cfg.read((char*)&this->iNy, sizeof(&(this->iNy)));
    cfg.read((char*)&this->iNz, sizeof(&(this->iNz)));
    cfg.read((char*)&this->rDx, sizeof(&(this->rDx)));
    cfg.read((char*)&this->rDy, sizeof(&(this->rDy)));
    cfg.read((char*)&this->rDz, sizeof(&(this->rDz)));

  }

  // Assign internal state
  this->iState = 0;

}


Cfg::~Cfg() {
}


int Cfg::Validate(void) {

  int iRetCode = 0; // Assume success (will falsify on failure)

  // Check "General" section validity
  if(this->iDebugLevel < 0) this->iDebugLevel = 0;
  if(this->iFrameInterval > 0) {
    if(this->sFramePath.empty()) {
      iRetCode = 1;
      this->iState = 0;
      return iRetCode;
    }
  }
  if(this->iExecMode < 0 || this->iExecMode > 1) {
    iRetCode = 2;
    this->iState = 0;
    return iRetCode;
  }

  // Check "Timing" section validity
  if(this->iAvgTime < 1 || this->iAvgTime > 3600) {
    iRetCode = 3;
    this->iState = 0;
    return iRetCode;
  }
  if(3600 % this->iAvgTime != 0) {
    iState = 4;
    this->iState = 0;
    return iRetCode;
  }
  if(this->iNumStep < 1 || this->iNumStep > this->iAvgTime) {
    iState = 5;
    this->iState = 0;
    return iRetCode;
  }
  if(this->iAvgTime % this->iNumStep != 0) {
    iState = 6;
    this->iState = 0;
    return iRetCode;
  }
  if(this->iNumPart < 1) {
    iState = 7;
    this->iState = 0;
    return iRetCode;
  }
  if(this->iMaxAge < this->iAvgTime) {
    iState = 8;
    this->iState = 0;
    return iRetCode;
  }
  if(this->iAvgTime % this->iMaxAge != 0) {
    iState = 9;
    this->iState = 0;
    return iRetCode;
  }

  // Check "Emission" section validity
  if(this->sStatic.empty() && this->sDynamic.empty()) {
    iState = 10;
    this->iState = 0;
    return iRetCode;
  }
  if(this->sStatic == this->sDynamic) {
    iState = 11;
    this->iState = 0;
    return iRetCode;
  }

  // Check "Meteo" section validity
  if(this->sMetInpFile.empty()) {
    iState = 12;
    this->iState = 0;
    return iRetCode;
  }
  if(this->rZ0 <= 0.0) {
    iState = 13;
    this->iState = 0;
    return iRetCode;
  }
  if(this->rZr <= 0.0) {
    iState = 14;
    this->iState = 0;
    return iRetCode;
  }
  if(this->rZt <= 0.0) {
    iState = 15;
    this->iState = 0;
    return iRetCode;
  }
  if(this->iHemisphere < 0 || this->iHemisphere > 1) {
    iState = 16;
    this->iState = 0;
    return iRetCode;
  }

  // Check "Output" section validity
  if(this->iNx < 1) {
    iState = 17;
    this->iState = 0;
    return iRetCode;
  }
  if(this->iNy < 1) {
    iState = 18;
    this->iState = 0;
    return iRetCode;
  }
  if(this->iNz < 1) {
    iState = 19;
    this->iState = 0;
    return iRetCode;
  }
  if(this->rDx <= 0.0) {
    iState = 20;
    this->iState = 0;
    return iRetCode;
  }
  if(this->rDy <= 0.0) {
    iState = 21;
    this->iState = 0;
    return iRetCode;
  }
  if(this->rDz <= 0.0) {
    iState = 22;
    this->iState = 0;
    return iRetCode;
  }

  // Leave: successful completion
  this->iState = 1;
  return iRetCode;

}


int Cfg::GetState(void) {
  return this->iState;
}
