! metpre.f90 - Pre-processor, reading data in ALAMO form and converting them to full profiled form
!              suitable for TinyPart use.
!
! Written by: Mauri Favaron
!

program Meteo_Preprocessor

	use Processing
	
	implicit none
	
	! Locals
	character(len=256)	:: sConfigFile
	integer				:: iRetCode
	type(Config)		:: tConfig
	
	! Get parameters
	if(command_argument_count() /= 1) then
		print *, "metpre - Meteorological preparer for TinyDisp"
		print *
		print *, "Usage:"
		print *
		print *, "  ./metpre <Config_File_Name>"
		print *
		print *, "Copyright 2019 by Servizi Territorio srl"
		print *, "This is open-source software, covered by MIT license."
		print *
		print *, "Written by: Mauri Favaron"
		print *
		stop 1
	end if
	call get_command_argument(1, sConfigFile)
	
	! Process data
	iRetCode = tConfig % read(10, 11, sConfigFile)
	if(iRetCode /= 0) then
		print *, "metpre:: error: Input file not opened - Return code = ", iRetCode
		stop
	end if

end program Meteo_Preprocessor
