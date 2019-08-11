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
public:
}

#endif
