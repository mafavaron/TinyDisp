NVCC=nvcc
NVCC_FLAGS = -g -G -Xcompiler --std=c++11 --expt-extended-lambda
LIBS = -lcurand

td.exe : td.cu ini.obj INIReader.obj
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

ini.obj : ini.c
	$(NVCC) -g -G -c ini.c -o ini.obj

INIReader.obj : INIReader.cpp ini.h
	$(NVCC) -g -G -c INIReader.cpp -o INIReader.obj

metpro : metpre.o meteo.o processing.o pbl_met.a
	pgfortran -o metpre metpre.o meteo.o processing.o pbl_met.a

metpre.o : metpre.f90 processing.mod pbl_met.mod
	pgfortran -c metpre.f90

processing.o processing.mod : processing.f90 meteo.mod pbl_met.mod
	pgfortran -c processing.f90

meteo.o meteo.mod : meteo.f90 pbl_met.mod
	pgfortran -c meteo.f90

clean :
	del *.mod
	del *.o
	del -f metpro
