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
    this->z.clear();
    this->u.clear();
    this->v.clear();
    this->w.clear();
    this->T.clear();
    this->su2.clear();
    this->sv2.clear();
    this->sw2.clear();
    this->dsw2.clear();
    this->eps.clear();
    this->alfa.clear();
    this->beta.clear();
    this->gamma.clear();
    this->delta.clear();
    this->alfa_u.clear();
    this->alfa_v.clear();
    this->deltau.clear();
    this->deltav.clear();
    this->deltat.clear();
    this->Au.clear();
    this->Av.clear();
    this->A.clear();
    this->B.clear();
    cfg.read((char*)&this->rEpoch, sizeof(double));
    for(int i=0; i<n; i++) {
      cfg.read((char*)&this->z[i], sizeof(double));
      cfg.read((char*)&this->u[i], sizeof(double));
      cfg.read((char*)&this->v[i], sizeof(double));
      this->w[i] = 0.0;
      cfg.read((char*)&this->T[i], sizeof(double));
      cfg.read((char*)&this->su2[i], sizeof(double));
      cfg.read((char*)&this->sv2[i], sizeof(double));
      cfg.read((char*)&this->sw2[i], sizeof(double));
      cfg.read((char*)&this->dsw2[i], sizeof(double));
      cfg.read((char*)&this->eps[i], sizeof(double));
      cfg.read((char*)&this->alfa[i], sizeof(double));
      cfg.read((char*)&this->beta[i], sizeof(double));
      cfg.read((char*)&this->gamma[i], sizeof(double));
      cfg.read((char*)&this->delta[i], sizeof(double));
      cfg.read((char*)&this->alfa_u[i], sizeof(double));
      cfg.read((char*)&this->alfa_v[i], sizeof(double));
      cfg.read((char*)&this->deltau[i], sizeof(double));
      cfg.read((char*)&this->deltav[i], sizeof(double));
      cfg.read((char*)&this->deltat[i], sizeof(double));
      cfg.read((char*)&this->Au[i], sizeof(double));
      cfg.read((char*)&this->Av[i], sizeof(double));
      cfg.read((char*)&this->A[i], sizeof(double));
      cfg.read((char*)&this->B[i], sizeof(double));
    }
  }

  return iRetCode;

};
