#include "meteodata.h"

MeteoData::MeteoData(int n) {
  this->z.reserve(n);
};


MeteoData::~MeteoData() {};


int MeteoData::Get(std::ifstream& cfg) {

  // Assume success (will falsify on failure)
  int iRetCode = 0;

  return iRetCode;

};
