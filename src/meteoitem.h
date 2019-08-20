#ifndef __METEOITEM__

class MeteoItem {
public:
  double rEpoch;	// Time stamp of current profile set
  double z;		    // Levels' height above ground (m)
  double u;		    // U components (m/s)
  double v;		    // V components (m/s)
  double w;		    // W components (m/s)
  double T;		    // Temperatures (K)
  double su2;		  // var(U) values (m2/s2)
  double sv2;		  // var(V) values (m2/s2)
  double sw2;		  // var(W) values (m2/s2)
  double dsw2;		// d var(W) / dz (m/s2)
  double eps;		  // TKE dissipation rate
  double alfa;		// Langevin equation coefficient
  double beta;		// Langevin equation coefficient
  double gamma;	  // Langevin equation coefficient
  double delta;	  // Langevin equation coefficient
  double alfa_u;	// Langevin equation coefficient
  double alfa_v;	// Langevin equation coefficient
  double deltau;	// Langevin equation coefficient
  double deltav;	// Langevin equation coefficient
  double deltat;	// Langevin equation coefficient
  double Au;		  // exp(alfa_u*dt)
  double Av;		  // exp(alfa_v*dt)
  double A;		    // exp(alfa*dt)
  double B;		    // exp(beta*dt)
};

#endif
