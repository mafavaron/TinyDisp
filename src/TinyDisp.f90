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
    character(len=256)                      :: sCfgFile
    type(ConfigType)                        :: tCfg
    type(MeteoType)                         :: tMeteo
    type(ParticlesPoolType)                 :: tPart
    integer                                 :: thread_id, nthreads
    integer                                 :: iRetCode
    integer                                 :: iMeteo
    integer                                 :: i
    integer                                 :: iNumActiveParticles
    integer, dimension(:,:), allocatable    :: imCount
    integer                                 :: iPartX
    integer                                 :: iPartY
	
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
    if(tCfg % iDebugLevel >= 1) print *, "Configuration read"
	
	! Read meteo data, and expand it to the desired time step
    iRetCode = tMeteo % read(10, tCfg % sMeteoFile)
    if(iRetCode /= 0) then
        print *, "TinyDisp:: Error: Meteorological file not read - Return code = ", iRetCode
        stop
    end if
    if(tCfg % iDebugLevel >= 1) print *, "Meteo file read"
    iRetCode = tMeteo % resample(tCfg % iTimeStep)
    if(iRetCode /= 0) then
        print *, "TinyDisp:: Error: Meteorological data not resampled - Return code = ", iRetCode
        stop
    end if
    if(tCfg % iDebugLevel >= 1) print *, "Meteo data resampled"
	
	! Initialize particles pool
    iRetCode = tPart % Create(tCfg % iMaxPart, tCfg % lTwoDimensional)
    if(iRetCode /= 0) then
        print *, "TinyDisp:: Error: Particle pool not initialized - Return code = ", iRetCode
        stop
    end if
    if(tCfg % iDebugLevel >= 1) print *, "Particle pool initialized"
    
    ! Initialize particles file
    open(10, file=tCfg % sParticlesFile, status='unknown', action='write', access='stream')
    if(tPart % lTwoDimensional) then
        write(10)  tCfg % iMaxPart, size(tMeteo % ivTimeStamp)
    else
        write(10) -tCfg % iMaxPart, size(tMeteo % ivTimeStamp)
    end if
    
    ! Initialize grid file, if requested
    if(tCfg % lEnableCounting) then
        open(11, file=tCfg % sCountingFile, status='unknown', action='write', access='stream')
        allocate(imCount(tCfg % iNumCells, tCfg % iNumCells))
        write(11) tCfg % rXmin, tCfg % rYmin, tCfg % rDxy, tCfg % iNumCells
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
        iRetCode = tPart % Move( &
            tMeteo % rvU(iMeteo), &
            tMeteo % rvV(iMeteo), &
            tMeteo % rvW(iMeteo), &
            tMeteo % rvStdDevU(iMeteo)**2, &
            tMeteo % rvStdDevV(iMeteo)**2, &
            tMeteo % rvStdDevW(iMeteo)**2, &
            tMeteo % rvCovUV(iMeteo), &
            tMeteo % rvCovUW(iMeteo), &
            tMeteo % rvCovVW(iMeteo), &
            float(tCfg % iTimeStep), &
            tCfg % rInertia &
        )
        
        ! Write particles
        iNumActiveParticles = count(tPart % ivTimeStampAtBirth >= 0)
        if(tPart % lTwoDimensional) then
            write(10) &
                iMeteo, &
                tMeteo % ivTimeStamp(iMeteo), &
                tMeteo % rvU(iMeteo), &
                tMeteo % rvV(iMeteo), &
                tMeteo % rvStdDevU(iMeteo)**2, &
                tMeteo % rvStdDevV(iMeteo)**2, &
                tMeteo % rvCovUV(iMeteo), &
                iNumActiveParticles
        else
            write(10) &
                iMeteo, &
                tMeteo % ivTimeStamp(iMeteo), &
                tMeteo % rvU(iMeteo), &
                tMeteo % rvV(iMeteo), &
                tMeteo % rvW(iMeteo), &
                tMeteo % rvStdDevU(iMeteo)**2, &
                tMeteo % rvStdDevV(iMeteo)**2, &
                tMeteo % rvStdDevW(iMeteo)**2, &
                tMeteo % rvCovUV(iMeteo), &
                tMeteo % rvCovUW(iMeteo), &
                tMeteo % rvCovVW(iMeteo), &
                iNumActiveParticles
        end if
        if(iNumActiveParticles > 0) then
            do i = 1, tCfg % iMaxPart
                if(tPart % ivTimeStampAtBirth(i) >= 0) then
                    write(10) tPart % rvX(i)
                end if
            end do
            do i = 1, tCfg % iMaxPart
                if(tPart % ivTimeStampAtBirth(i) >= 0) then
                    write(10) tPart % rvY(i)
                end if
            end do
            if(.not. tPart % lTwoDimensional) then
                do i = 1, tCfg % iMaxPart
                    if(tPart % ivTimeStampAtBirth(i) >= 0) then
                        write(10) tPart % rvZ(i)
                    end if
                end do
            end if
            do i = 1, tCfg % iMaxPart
                if(tPart % ivTimeStampAtBirth(i) >= 0) then
                    write(10) tPart % ivTimeStampAtBirth(i)
                end if
            end do
        end if
        
        ! If requested, generate and save gridded counts
        if(tCfg % lEnableCounting) then
            imCount = 0
            do i = 1, tCfg % iMaxPart
                if(tPart % ivTimeStampAtBirth(i) >= 0) then
                    iPartX = floor((tPart % rvX(i) - tCfg % rXmin) / tCfg % rDxy) + 1
                    iPartY = floor((tPart % rvY(i) - tCfg % rYmin) / tCfg % rDxy) + 1
                    if(1 <= iPartX .and. iPartX <= tCfg % iNumCells .and. 1 <= iPartY .and. iPartY <= tCfg % iNumCells) then
                        imCount(iPartX,iPartY) = imCount(iPartX,iPartY) + 1
                    end if
                end if
            end do
            write(11) tMeteo % ivTimeStamp(iMeteo)
            do i = 1, size(imCount, dim=2)
                write(11) imCount(:,i)
            end do
        end if

        ! Inform users of progress
        if(tCfg % iDebugLevel >= 2) print *, "Step: ", iMeteo, " of ", size(tMeteo % ivTimeStamp), " - P: ", iNumActiveParticles
        
    end do

    ! Release connection with grid file, if requested
    if(tCfg % lEnableCounting) then
        deallocate(imCount)
        close(11)
    end if
	
end program TinyDisp
