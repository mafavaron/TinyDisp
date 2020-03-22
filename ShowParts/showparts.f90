program showparts

    use dislin
    use config
    use data_file

    implicit none
    
    ! Locals
    integer             :: iRetCode
    character(len=256)  :: sConfigFile
    character(len=256)  :: sInputFile
    character(len=256)  :: sMovieFile
    type(PartType)      :: tPart
    type(ConfigType)    :: tCfg
    integer             :: iNumIter
    integer             :: iCountTotal
    integer             :: iMinTimeStamp
    
    integer, parameter  :: n = 100
    integer             :: i
    integer             :: ic
    real, dimension(n)  :: xray, yray1, yray2
    
    xray = [((i-1)*360./99., i = 1, 100)]
    yray1 = sin(xray * 3.1415927/180.)
    yray2 = cos(xray * 3.1415927/180.)
    
    ! Check input parameters
    if(command_argument_count() /= 2) then
        print *, "showparts - Movie producer for TinyDisp outputs"
        print *
        print *, "Usage:"
        print *
        print *, "  showparts <ConfigFile> <MovieFile>"
        print *
        print *, "Copyright 2020 by Servizi Territorio srl"
        print *, "This is open-source software, covered by the MIT license"
        print *
        stop
    end if
    call get_command_argument(1, sConfigFile)
    call get_command_argument(2, sMovieFile)

    ! Get configuration
    iRetCode = tCfg % Read(10, sConfigFile)
    if(iRetCode /= 0) then
        print *, "Error: Configuration not read - Return code = ", iRetCode, " - Cfg file: ", trim(sConfigFile)
        stop
    end if
    
    ! Start accessing particles file
    iRetCode = tPart % Open(10, tCfg % sParticlesFile)
    if(iRetCode /= 0) then
        print *, "Error: Configuration not read"
        stop
    end if
    
    ! Start DISLIN
    ! Main loop: get particles, and inspect them
    iNumIter = 0
    do
    
        iNumIter = iNumIter + 1
    
        ! Actual read attempt
        iRetCode = tPart % Read()
        if(iRetCode /= 0) exit
        
        ! Check the number of particles
        iCountTotal = tPart % iNumPart
        if(iCountTotal > 0) then
            iMinTimeStamp = minval(tPart % ivTimeStamp)
        else
            iMinTimeStamp = 0
        end if
        
        call METAFL('PNG')
        call SCRMOD('REVERS')
        call DISINI()
        call WINSIZ(800, 800)
        call HSYMBL(3)
        call COLOR('RED')
        call QPLSCA(yray1, yray2, n)
        call DISFIN()
        
        ! Inform users
        print "(1x,2(i10,','),i10)", iNumIter, iCountTotal, iMinTimeStamp
        
    end do

    ! Leave
    iRetCode = tPart % Close()
    
end program showparts
