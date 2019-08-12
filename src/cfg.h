#ifndef __CFG__

#include <string>

class Cfg {
private:
  // Internal status
  int         iState;
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
  Cfg();
  Cfg(const std::string& sCfgFileName);
  ~Cfg();
  int Validate(void);
};

#endif
