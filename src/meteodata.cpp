#include "meteodata.h"

MeteoData::MeteoData(const int n) {
  this->z.reserve(n);
  this->u.reserve(n);
  this->v.reserve(n);
  this->w.reserve(n);
  this->T.reserve(n);
  this->su2.reserve(n);
  this->sv2.reserve(n);
  this->sw2.reserve(n);
  this->dsw2.reserve(n);
  this->eps.reserve(n);
  this->alfa.reserve(n);
  this->beta.reserve(n);
  this->gamma.reserve(n);
  this->delta.reserve(n);
  this->alfa_u.reserve(n);
  this->alfa_v.reserve(n);
  this->deltau.reserve(n);
  this->deltav.reserve(n);
  this->deltat.reserve(n);
  this->Au.reserve(n);
  this->Av.reserve(n);
  this->A.reserve(n);
  this->B.reserve(n);
};


MeteoData::~MeteoData() {};


int MeteoData::Get(std::ifstream& cfg, const int n) {

  // Assume success (will falsify on failure)
  int iRetCode = 0;

  // Get oen record< if present
  if(cfg.is_open()) {
    for(int i=0; i<n; i++) {
      cfg.read((char*)&this->iDebugLevel, sizeof(int));
    }
  }

  return iRetCode;

};
