Test_NormalDeviates.exe : Test_NormalDeviates.obj NormalDeviates.obj
	opgfortran -o Test_NormalDeviates.exe Test_NormalDeviates.obj NormalDeviates.obj

Meteo_Test.exe : Meteo_Test.obj Meteo.obj Calendar.obj
	opgfortran -o Meteo_Test.exe Meteo_Test.obj Meteo.obj Calendar.obj

Meteo_Test.obj : Meteo_Test.f90 Calendar.mod Meteo.mod
	pgfortran -c Meteo_Test.f90
	
Meteo.obj Meteo.mod : Meteo.f90 Calendar.mod
	pgfortran -c Meteo.f90

Calendar.obj Calendar.mod : Calendar.f90
	pgfortran -c Calendar.f90

Test_NormalDeviates.obj : Test_NormalDeviates.f90
	pgfortran -c Test_NormalDeviates.f90

NormalDeviates.obj NormalDeviates.mod : NormalDeviates.f90
	pgfortran -c NormalDeviates.f90

