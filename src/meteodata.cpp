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


int MeteoData::Read(std::ifstream& cfg, const int n) {

  // Assume success (will falsify on failure)
  int iRetCode = 0;

  // Get oen record< if present
  if(cfg.peek() != EOF) {
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
  else {
    iRetCode = -1; // End-of-file, a special case of successful return
  }

  return iRetCode;

};


double MeteoData::GetTimeStamp() {
  return this->rEpoch;
}


int MeteoData::Evaluate(const float rReferenceZ, const double rZ0, const double rDz, MeteoItem* tMet) {

  // Assume success (will falsify on failure
  int iRetCode = 0;

  // Identify the indices bounding the desired height
  int n = this->z.size();
  int izFrom;
  int izTo;
  if(rReferenceZ <= this->z[0]) {
    izFrom = 0;
    izTo   = 0;
  }
  else if(rReferenceZ >= this->z[n-1]) {
    izFrom = n - 1;
    izTo   = n - 1;
  }
  else {  // Entry condition: z(1) < zp < z(n)
    izFrom = floor((rReferenceZ - rZ0) / rDz);
    izTo   = ceil((rReferenceZ - rZ0) / rDz);
    if(izFrom < 0) izFrom = 0;
    if(izFrom > n-1) izFrom = n-1;
  }

  // Evaluate linear interpolation coefficients
  double zpp = (rReferenceZ - this->z[izFrom]) / rDz;

  // Compute linear interpolation
  tMet->u      = this->u[izFrom]      + zpp * (this->u[izTo]      - this->u[izFrom]);
  tMet->v      = this->v[izFrom]      + zpp * (this->v[izTo]      - this->v[izFrom]);
  tMet->su2    = this->su2[izFrom]    + zpp * (this->su2[izTo]    - this->su2[izFrom]);
  tMet->sv2    = this->sv2[izFrom]    + zpp * (this->sv2[izTo]    - this->sv2[izFrom]);
  tMet->sw2    = this->sw2[izFrom]    + zpp * (this->sw2[izTo]    - this->sw2[izFrom]);
  tMet->dsw2   = this->dsw2[izFrom]   + zpp * (this->dsw2[izTo]   - this->dsw2[izFrom]);
  tMet->eps    = this->eps[izFrom]    + zpp * (this->eps[izTo]    - this->eps[izFrom]);
  tMet->alfa   = this->alfa[izFrom]   + zpp * (this->alfa[izTo]   - this->alfa[izFrom]);
  tMet->beta   = this->beta[izFrom]   + zpp * (this->beta[izTo]   - this->beta[izFrom]);
  tMet->gamma  = this->gamma[izFrom]  + zpp * (this->gamma[izTo]  - this->gamma[izFrom]);
  tMet->delta  = this->delta[izFrom]  + zpp * (this->delta[izTo]  - this->delta[izFrom]);
  tMet->alfa_u = this->alfa_u[izFrom] + zpp * (this->alfa_u[izTo] - this->alfa_u[izFrom]);
  tMet->alfa_v = this->alfa_v[izFrom] + zpp * (this->alfa_v[izTo] - this->alfa_v[izFrom]);
  tMet->deltau = this->deltau[izFrom] + zpp * (this->deltau[izTo] - this->deltau[izFrom]);
  tMet->deltav = this->deltav[izFrom] + zpp * (this->deltav[izTo] - this->deltav[izFrom]);
  tMet->deltat = this->deltat[izFrom] + zpp * (this->deltat[izTo] - this->deltat[izFrom]);
  tMet->Au     = this->Au[izFrom]     + zpp * (this->Au[izTo]     - this->Au[izFrom]);
  tMet->Av     = this->Av[izFrom]     + zpp * (this->Av[izTo]     - this->Av[izFrom]);
  tMet->A      = this->A[izFrom]      + zpp * (this->A[izTo]      - this->A[izFrom]);
  tMet->B      = this->B[izFrom]      + zpp * (this->B[izTo]      - this->B[izFrom]);

  // Leave
  return iRetCode;

}
