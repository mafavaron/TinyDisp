program testConfig

	use Config
	
	implicit none
	
	type(ConfigType)	:: tCfg
	integer				:: iRetCode
	integer				:: i
    integer, dimension(:), allocatable  :: ivTimeStamp
    real, dimension(:), allocatable     :: rvU
    real, dimension(:), allocatable     :: rvV
    real, dimension(:), allocatable     :: rvStdDevU
    real, dimension(:), allocatable     :: rvStdDevV
    real, dimension(:), allocatable     :: rvCovUV
	
	iRetCode = tCfg % gather(10, '.\\test.nml')
    if(iRetCode /= 0) then
        print *, "Error reading configuration - Ret code = ", iRetCode
        stop
    end if
    
    iRetCode = tCfg % get_meteo(ivTimeStamp, rvU, rvV, rvStdDevU, rvStdDevV, rvCovUV)
    if(iRetCode /= 0) then
        print *, "Error gathering meteo sample - Ret code = ", iRetCode
        stop
    end if
    
    !do i = 1, size(ivTimeStamp)
    !    print *, ivTimeStamp(i), rvU(i), rvV(i)
    !end do
    
    print *, tCfg % get_num_particles()
    
end program testConfig

