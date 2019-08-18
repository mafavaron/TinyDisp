#ifndef __METEODATA__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class MeteoData {
private:
  // Time stamp
  double				      rEpoch;	// Time stamp of current profile set
  // Primitive profiles
  std::vector<double> z;		  // Levels' height above ground (m)
  std::vector<double> u;		  // U components (m/s)
  std::vector<double> v;		  // V components (m/s)
  std::vector<double> T;		  // Temperatures (K)
  std::vector<double> su2;		// var(U) values (m2/s2)
  std::vector<double> sv2;		// var(V) values (m2/s2)
  std::vector<double> sw2;		// var(W) values (m2/s2)
  std::vector<double> dsw2;		// d var(W) / dz (m/s2)
  std::vector<double> eps;		// TKE dissipation rate
  std::vector<double> alfa;		// Langevin equation coefficient
  std::vector<double> beta;		// Langevin equation coefficient
  std::vector<double> gamma;	// Langevin equation coefficient
  std::vector<double> delta;	// Langevin equation coefficient
  std::vector<double> alfa_u;	// Langevin equation coefficient
  std::vector<double> alfa_v;	// Langevin equation coefficient
  std::vector<double> deltau;	// Langevin equation coefficient
  std::vector<double> deltav;	// Langevin equation coefficient
  std::vector<double> deltat;	// Langevin equation coefficient
  // Convenience derived values
  std::vector<double> Au;		 // exp(alfa_u*dt)
  std::vector<double> Av;		 // exp(alfa_v*dt)
  std::vector<double> A;		 // exp(alfa*dt)
  std::vector<double> B;		 // exp(beta*dt)
public:
  MeteoData(int n);
  ~MeteoData();
  int Get(std::ifstream& cfg);
};

#endif
