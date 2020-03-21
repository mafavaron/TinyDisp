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
    yray1 = sin(xray)
    yray2 = cos(xray)
    
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
    call METAFL('CONS')
    call SCRMOD('REVERS')
    call DISINI()
    call PAGERA()
    call COMPLX()
    call AXSPOS(450, 1800)
    call AXSLEN(2200, 1200)
    call NAME('X-axis', 'X')
    call NAME('Y-axis', 'Y')
    call LABDIG(-1, 'X')
    call TICKS(10, 'XY')
    call TITLIN('Curve(s)', 1)
    call TITLIN('SIN(X), COS(X)', 3)
    ic = INTRGB(0.95, 0.95, 0.95)
    call AXSBGD(ic)
    call GRAF(0., 360., 0., 90., -1., 1., -1., 0.5)
    call SETRGB(0.7, 0.7, 0.7)
    call GRID(1,1)
    call COLOR('FORE')
    call TITLE()
    call COLOR('RED')
    call CURVE(xray, yray1, n)
    call COLOR('BLUE')
    call CURVE(xray, yray2, n)
    
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
        
        ! Inform users
        print "(1x,2(i10,','),i10)", iNumIter, iCountTotal, iMinTimeStamp
        
    end do

    ! Leave
    call DISFIN()
    iRetCode = tPart % Close()
    
end program showparts
