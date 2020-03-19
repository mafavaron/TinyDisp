! Program 'ptest' - Calculate some useful particles statistics
!
! Copyright 2020 by Servizi Territorio srl
!                   All rights reserved
!
! By: Patrizia M. Favaron
!
program ptest

    use config
    use data_file

    implicit none
    
    type(ConfigType)    :: tCfg
    type(PartType)      :: tPart
    integer             :: iRetCode
    character(len=256)  :: sIniFileName
    real                :: rMeanX
    real                :: rMeanY
    real                :: rMinX
    real                :: rMinY
    real                :: rMaxX
    real                :: rMaxY
    integer             :: iCountTotal
    integer             :: iCountInside
    
    ! Get parameters
    if(command_argument_count() /= 1) then
        print *, "showparts - Application to visualize modelling results"
        print *
        print *, "Usage:"
        print *
        print *, "    showparts <IniFile>"
        print *
        print *, "Copyright 2020 by Servizi Territorio srl"
        print *, "                  All rights reserved"
        stop 1
    end if
    call get_command_argument(1, sIniFileName)
    
    ! Get configuration
    iRetCode = tCfg % Read(10, sIniFileName)
    if(iRetCode /= 0) then
        print *, "Error: Configuration not read"
        stop
    end if
    
    ! Start accessing particles file
    iRetCode = tPart % Open(10, tCfg % sParticlesFile)
    if(iRetCode /= 0) then
        print *, "Error: Configuration not read"
        stop
    end if
    
    ! Main loop: get particles, and inspect them
    do
    
        ! Actual read attempt
        iRetCode = tPart % Read()
        if(iRetCode /= 0) exit
        
        ! Check the number of particles
        iCountTotal = tPart % iNumPart
        if(iCountTotal > 0) then
            rMeanX = sum(tPart % rvX(1:iCountTotal)) / iCountTotal
            rMeanY = sum(tPart % rvY(1:iCountTotal)) / iCountTotal
            rMinX  = minval(tPart % rvX(1:iCountTotal))
            rMinY  = minval(tPart % rvY(1:iCountTotal))
            rMaxX  = maxval(tPart % rvX(1:iCountTotal))
            rMaxY  = maxval(tPart % rvY(1:iCountTotal))
            iCountInside = count(abs(tPart % rvX) <= tCfg % rEdgeLength/2. .and. abs(tPart % rvY) <= tCfg % rEdgeLength/2.)
        else
            rMeanX = -9999.9
            rMeanY = -9999.9
            rMinX  = -9999.9
            rMinY  = -9999.9
            rMaxX  = -9999.9
            rMaxY  = -9999.9
            iCountInside = 0
        end if
        
        ! Inform users
        print "(1x,6(f8.2,', '),i10)", rMinX, rMeanX, rMaxX, rMinY, rMeanY, rMaxY, iCountInside
        
    end do

    ! Leave
    iRetCode = tPart % Close()
    
end program ptest
