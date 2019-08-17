#ifndef __METEO__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class Meteo {
private:
  // Time stamp
  double				 rEpoch	// Time stamp of current profile set
  // Primitive profiles
  vector<double> z		  // Levels' height above ground (m)
  vector<double> u		  // U components (m/s)
  vector<double> v		  // V components (m/s)
  vector<double> T		  // Temperatures (K)
  vector<double> su2		// var(U) values (m2/s2)
  vector<double> sv2		// var(V) values (m2/s2)
  vector<double> sw2		// var(W) values (m2/s2)
  vector<double> dsw2		// d var(W) / dz (m/s2)
  vector<double> eps		// TKE dissipation rate
  vector<double> alfa		// Langevin equation coefficient
  vector<double> beta		// Langevin equation coefficient
  vector<double> gamma	// Langevin equation coefficient
  vector<double> delta	// Langevin equation coefficient
  vector<double> alfa_u	// Langevin equation coefficient
  vector<double> alfa_v	// Langevin equation coefficient
  vector<double> deltau	// Langevin equation coefficient
  vector<double> deltav	// Langevin equation coefficient
  vector<double> deltat	// Langevin equation coefficient
  // Convenience derived values
  vector<double> Au		 // exp(alfa_u*dt)
  vector<double> Av		 // exp(alfa_v*dt)
  vector<double> A		 // exp(alfa*dt)
  vector<double> B		 // exp(beta*dt)
public:
  Meteo(int n);
  ~Meteo();
  Get(std::ifstream& cfg);
};

#endif
