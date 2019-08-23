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
    this->sDiaFile = this->sDiaFile.erase(this->sDiaFile.find_last_not_of(" \n\r\t")+1);
    cfg.read((char*)&this->iFrameInterval, sizeof(int));
    cfg.read(buffer, 256);
    this->sFramePath = buffer;
    this->sFramePath = this->sFramePath.erase(this->sFramePath.find_last_not_of(" \n\r\t")+1);
    cfg.read((char*)&this->iExecMode, sizeof(int));

    // Timing
    cfg.read((char*)&this->iAvgTime, sizeof(int));
    cfg.read((char*)&this->iNumStep, sizeof(int));
    cfg.read((char*)&this->iNumPart, sizeof(int));
    cfg.read((char*)&this->iMaxAge, sizeof(int));

    // Emission
    cfg.read(buffer, 256);
    this->sStatic = buffer;
    this->sStatic = this->sStatic.erase(this->sStatic.find_last_not_of(" \n\r\t")+1);
    cfg.read(buffer, 256);
    this->sDynamic = buffer;
    this->sDynamic = this->sDynamic.erase(this->sDynamic.find_last_not_of(" \n\r\t")+1);

    // Meteo
    cfg.read(buffer, 256);
    this->sMetInpFile = buffer;
    this->sMetInpFile = this->sMetInpFile.erase(this->sMetInpFile.find_last_not_of(" \n\r\t")+1);
    cfg.read(buffer, 256);
    this->sMetOutFile = buffer;
    this->sMetOutFile = this->sMetOutFile.erase(this->sMetOutFile.find_last_not_of(" \n\r\t")+1);
    cfg.read(buffer, 256);
    this->sMetDiaFile = buffer;
    this->sMetDiaFile = this->sMetDiaFile.erase(this->sMetDiaFile.find_last_not_of(" \n\r\t")+1);
    cfg.read((char*)&this->rHeight, sizeof(double));
    cfg.read((char*)&this->rZ0, sizeof(double));
    cfg.read((char*)&this->rZr, sizeof(double));
    cfg.read((char*)&this->rZt, sizeof(double));
    cfg.read((char*)&this->rGamma, sizeof(double));
    cfg.read((char*)&this->iHemisphere, sizeof(int));

    // Output
    cfg.read(buffer, 256);
    this->sConcFile = buffer;
    this->sConcFile = this->sConcFile.erase(this->sConcFile.find_last_not_of(" \n\r\t")+1);
    cfg.read((char*)&this->rX0, sizeof(double));
    cfg.read((char*)&this->rY0, sizeof(double));
    cfg.read((char*)&this->iNx, sizeof(int));
    cfg.read((char*)&this->iNy, sizeof(int));
    cfg.read((char*)&this->iNz, sizeof(int));
    cfg.read((char*)&this->rDx, sizeof(double));
    cfg.read((char*)&this->rDy, sizeof(double));
    cfg.read((char*)&this->rDz, sizeof(double));
    cfg.read((char*)&this->rFactor, sizeof(double));

  }
  else {
    this->iState = 0;
  }

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


int Cfg::GetNumZ(void) {
  return this->iNz;
}


double Cfg::GetDomainCenterX(void) {
  return this->rX0 + (this->rDx * this->iNx) / 2.0;
}


double Cfg::GetDomainCenterY(void) {
  return this->rY0 + (this->rDy * this->iNy) / 2.0;
}


int Cfg::GetPartPoolSize(void) {
  return this->iMaxAge * this->iNumPart / this->iNumStep;
}


double Cfg::GetX0(void) {
  return this->rX0;
}


double Cfg::GetDx(void) {
  return this->rDx;
}


int Cfg::GetNx(void) {
  return this->iNx;
}


double Cfg::GetZ0(void) {
  return this->rZ0;
}


double Cfg::GetDz(void) {
  return this->rDz;
}


int Cfg::GetNz(void) {
  return this->iNz;
}


int Cfg::GetPartToEmitPerSource(void) {
  return this->iNumPart;
}


int Cfg::GetTimeSubstepDuration(void) {
  return this->iAvgTime / this->iNumStep;
}
