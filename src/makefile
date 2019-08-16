NVCC=nvcc
NVCC_FLAGS = -g -G -Xcompiler --std=c++11 --expt-extended-lambda
LIBS = -lcurand

td.exe : td.cu cfg.obj
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

cfg.obj : cfg.cpp
	$(NVCC) -g -G -c cfg.cpp -o cfg.obj

metpre.exe : metpre.obj meteo.obj processing.obj pbl_met.lib
	pgfortran -o metpre.exe metpre.obj meteo.obj processing.obj pbl_met.lib

metpre.obj : metpre.f90 processing.mod pbl_met.mod
	pgfortran -c metpre.f90

processing.obj processing.mod : processing.f90 meteo.mod pbl_met.mod
	pgfortran -c processing.f90

meteo.obj meteo.mod : meteo.f90 pbl_met.mod
	pgfortran -c meteo.f90

clean :
	del *.mod
	del *.obj
	del -f metpro.exe
	del -f td.exe
