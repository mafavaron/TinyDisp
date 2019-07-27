metpro : metpro.o meteo.o processing.o pbl_met.a
	gfortran -o metpro metpro.o meteo.o processing.o pbl_met.a
	
metpro.o : metpro.f90 processing.mod pbl_met.mod
	gfortran -c metpro.f90
	
processing.o processing.mod : processing.f90 meteo.mod pbl_met.mod
	gfortran -c processing.f90
	
meteo.o meteo.mod : meteo.f90 pbl_met.mod
	gfortran -c meteo.f90

clean :
	rm *.mod
	rm *.o
	rm -f metpro
	
