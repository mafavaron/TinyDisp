#ifndef __CFG__

class Cfg {
private:
  // General
  int         iDebugLevel;
  std::string sDiaFile;
  int         iFrameInterval;
  std::string sFramePath;
  int         iExecMode;
  // Timing
  int         iAvgTime;
  int         iNumStep;
  int         iNumPart;
  int         iMaxAge;
  // Emission
  std::string sStatic;
  std::string sDynamic;
  // Meteo
  std::string sMetInpFile;
  std::string sMetOutFile;
  std::string sMetDiaFile;
  double      rHeight;
  double      rZ0;
  double      rZr;
  double      rZt;
  double      rGamma;
  int         iHemisphere;
  // Output
  std::string sConcFile;
  double      rX0;
  double      rY0;
  int         iNx;
  int         iNy;
  double      rDx;
  double      rDy;
  int         iNz;
  double      rDz;
public:
}

#endif
