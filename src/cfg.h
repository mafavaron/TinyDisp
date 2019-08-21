#ifndef __CFG__
#define __CFG__

#include <iostream>
#include <fstream>
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
  double      rFactor;
public:
  // Constructors, destructor
  Cfg();
  Cfg(std::ifstream& cfg);
  ~Cfg();
  // Parameter check
  int Validate(void);
  // Access to components
  int GetState(void);
  int GetNumZ(void);
  double GetDomainCenterX(void);
  double GetDomainCenterY(void);
  int GetPartPoolSize(void);
  double GetZ0(void);
  double GetDz(void);
  int GetPartToEmitPerSource(void);
  int GetTimeSubstepDuration(void);
};

#endif
