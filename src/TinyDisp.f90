! Main program of TinyDisp
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program TinyDisp

    use omp_lib
    use Config
    use Meteo
    use Particles
	
    implicit none
	
	! Locals
    character(len=256)      :: sCfgFile
    type(ConfigType)        :: tCfg
    type(MeteoType)         :: tMeteo
    type(ParticlesPoolType) :: tPart
    integer                 :: thread_id, nthreads
    integer                 :: iRetCode
    integer                 :: iMeteo
	
	! Get input parameters
    if(command_argument_count() /= 1) then
        print *, "TinyDisp - Main processing module"
        print *
        print *, "Usage:"
        print *
        print *, "  TinyDisp <Configuration_File_Name>"
        print *
        print *, "Copyright 2020 by Servizi Territorio srl"
        print *, "                  This is open-source code, covered by the MIT license"
        print *
        stop
    end if
    call get_command_argument(1, sCfgFile)
	
	! Get configuration
    iRetCode = tCfg % get(10, sCfgFile)
    if(iRetCode /= 0) then
        print *, "TinyDisp:: Error: Configuration file not read - Return code = ", iRetCode
        stop
    end if
	
	! Read meteo data, and expand it to the desired time step
    iRetCode = tMeteo % read(10, tCfg % sMeteoFile)
    if(iRetCode /= 0) then
        print *, "TinyDisp:: Error: Meteorological file not read - Return code = ", iRetCode
        stop
    end if
    iRetCode = tMeteo % resample(tCfg % iTimeStep)
    if(iRetCode /= 0) then
        print *, "TinyDisp:: Error: Meteorological data not resampled - Return code = ", iRetCode
        stop
    end if
	
	! Initialize particles pool
    iRetCode = tPart % Create(tCfg % iMaxPart, tCfg % lTwoDimensional)
    if(iRetCode /= 0) then
        print *, "TinyDisp:: Error: Particle pool not initialized - Return code = ", iRetCode
        stop
    end if
	
	! Main loop: iterate over all time steps, and simulate transport and diffusion
    do iMeteo = 1, size(tMeteo % ivTimeStamp)
    
        ! Release new particles
        iRetCode = tPart % Emit( &
            tCfg % iNumPartsEmittedPerStep, &
            tMeteo % ivTimeStamp(iMeteo), &
            tMeteo % rvU(iMeteo), &
            tMeteo % rvV(iMeteo), &
            tMeteo % rvW(iMeteo)  &
        )
        if(iRetCode /= 0) then
            print *, "TinyDisp:: Error: Particles not emitted - Return code = ", iRetCode
            stop
        end if
        
        ! Move particles
        
    end do

    !$omp parallel private(thread_id)

    thread_id = omp_get_thread_num()
    write (*,*) 'Hello World from thread', thread_id

    !$omp barrier
    if ( thread_id == 0 ) then
        nthreads = omp_get_num_threads()
        write (*,*) 'There are', nthreads, 'threads'
    end if
    
    !$omp end parallel
	
		! Emit new particles
		
		! Move particles
		
		! Save active particles

end program TinyDisp
