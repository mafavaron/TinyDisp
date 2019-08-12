#include "cfg.h"
#include "INIReader.h"

#include <string>

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


Cfg::Cfg(const std::string& sCfgFileName) {

  // Read configuration
  INIReader cfg = INIReader(sCfgFileName);
  // -1- General
  std::string sSection = "General";
  std::string sName    = "debug_level";
  this->iDebugLevel = cfg.GetInteger(sSection, sName, 4);
  sName = "diafile";
  std::string sDefault = "";
  this->sDiaFile = cfg.GetString(sSection, sName, sDefault);
  sName = "frame_interval";
  this->iFrameInterval = cfg.GetInteger(sSection, sName, 0);
  sName = "frame_path";
  this->sFramePath = cfg.GetString(sSection, sName, sDefault);
  sName = "exec_mode";
  this->iExecMode = cfg.GetInteger(sSection, sName, 0);
  // -1- Timing
  sSection = "Timing";
  sName = "avgtime";
  this->iAvgTime = cfg.GetInteger(sSection, sName, 0);
  sName = "nstep";
  this->iNumStep = cfg.GetInteger(sSection, sName, 0);
  sName = "npart";
  this->iNumPart = cfg.GetInteger(sSection, sName, 0);
  sName = "maxage";
  this->iMaxAge = cfg.GetInteger(sSection, sName, 0);
  // -1- Emission
  sSection = "Emission";
  sName = "static";
  this->sStatic = cfg.GetString(sSection, sName, sDefault);
  sName = "dynamic";
  this->sDynamic = cfg.GetString(sSection, sName, sDefault);
  // -1- Meteo
  sSection = "Meteo";
  sName = "inpfile";
  this->sMetInpFile = cfg.GetString(sSection, sName, sDefault);
  sName = "outfile";
  this->sMetOutFile = cfg.GetString(sSection, sName, sDefault);
  sName = "diafile";
  this->sMetDiaFile = cfg.GetString(sSection, sName, sDefault);
  sName = "height";
  this->rHeight = cfg.GetReal(sSection, sName, 0.0);
  sName = "z0";
  this->rZ0 = cfg.GetReal(sSection, sName, 0.0);
  sName = "zr";
  this->rZr = cfg.GetReal(sSection, sName, 0.0);
  sName = "zt";
  this->rZt = cfg.GetReal(sSection, sName, 0.0);
  sName = "gamma";
  this->rGamma = cfg.GetReal(sSection, sName, 0.0);
  sName = "hemisphere";
  this->iHemisphere = cfg.GetInteger(sSection, sName, 0);
  // -1- Output
  sSection = "Output";
  sName = "conc";
  this->sConcFile = cfg.GetString(sSection, sName, sDefault);
  sName = "x0";
  this->rX0 = cfg.GetReal(sSection, sName, 0.0);
  sName = "y0";
  this->rY0 = cfg.GetReal(sSection, sName, 0.0);
  sName = "nx";
  this->iNx = cfg.GetInteger(sSection, sName, 0);
  sName = "ny";
  this->iNy = cfg.GetInteger(sSection, sName, 0);
  sName = "dx";
  this->rDx = cfg.GetReal(sSection, sName, 0.0);
  sName = "dy";
  this->rDy = cfg.GetReal(sSection, sName, 0.0);
  sName = "nz";
  this->iNz = cfg.GetInteger(sSection, sName, 0);
  sName = "dz";
  this->rDz = cfg.GetReal(sSection, sName, 0.0);

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
