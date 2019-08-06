td : td.o ini.o INIReader.o
	nvcc -o td td.o ini.o INIReader.o
	
td.o : td.cu ini.h INIReader.h
	nvcc -c td.cu

INIReader.o : INIReader.cu ini.h
	nvcc -c INIReader.cu

ini.o : ini.cu
	nvcc -c ini.cu

metpro : metpre.o meteo.o processing.o pbl_met.a
	gfortran -o metpre metpre.o meteo.o processing.o pbl_met.a
	
metpre.o : metpre.f90 processing.mod pbl_met.mod
	gfortran -c metpre.f90
	
processing.o processing.mod : processing.f90 meteo.mod pbl_met.mod
	gfortran -c processing.f90
	
meteo.o meteo.mod : meteo.f90 pbl_met.mod
	gfortran -c meteo.f90

clean :
	rm *.mod
	rm *.o
	rm -f metpro
	
