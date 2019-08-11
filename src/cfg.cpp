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
