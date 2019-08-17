#ifndef __METEO__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class Meteo {
private:
  ! Time stamp
  real(8)								:: rEpoch	! Time stamp of current profile set
  ! Primitive profiles
  real(8), dimension(:), allocatable	:: z		! Levels' height above ground (m)
  real(8), dimension(:), allocatable	:: u		! U components (m/s)
  real(8), dimension(:), allocatable	:: v		! V components (m/s)
  real(8), dimension(:), allocatable	:: T		! Temperatures (K)
  real(8), dimension(:), allocatable	:: su2		! var(U) values (m2/s2)
  real(8), dimension(:), allocatable	:: sv2		! var(V) values (m2/s2)
  real(8), dimension(:), allocatable	:: sw2		! var(W) values (m2/s2)
  real(8), dimension(:), allocatable	:: dsw2		! d var(W) / dz (m/s2)
  real(8), dimension(:), allocatable	:: eps		! TKE dissipation rate
  real(8), dimension(:), allocatable	:: alfa		! Langevin equation coefficient
  real(8), dimension(:), allocatable	:: beta		! Langevin equation coefficient
  real(8), dimension(:), allocatable	:: gamma	! Langevin equation coefficient
  real(8), dimension(:), allocatable	:: delta	! Langevin equation coefficient
  real(8), dimension(:), allocatable	:: alfa_u	! Langevin equation coefficient
  real(8), dimension(:), allocatable	:: alfa_v	! Langevin equation coefficient
  real(8), dimension(:), allocatable	:: deltau	! Langevin equation coefficient
  real(8), dimension(:), allocatable	:: deltav	! Langevin equation coefficient
  real(8), dimension(:), allocatable	:: deltat	! Langevin equation coefficient
  ! Convenience derived values
  real(8), dimension(:), allocatable	:: Au		! exp(alfa_u*dt)
  real(8), dimension(:), allocatable	:: Av		! exp(alfa_v*dt)
  real(8), dimension(:), allocatable	:: A		! exp(alfa*dt)
  real(8), dimension(:), allocatable	:: B		! exp(beta*dt)
public:
  procedure	:: clean      => metpClean
  procedure	:: alloc      => metpAlloc
  procedure	:: initialize => metpInitialize
  procedure	:: create     => metpCreate
  procedure	:: evaluate   => metpEvaluate
  procedure	:: dump       => metpDump
};

#endif
