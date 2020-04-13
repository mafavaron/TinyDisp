! Main program of dia_part
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program dia_part

    use TinyDispFiles
    use calendar
    
    implicit none
    
    ! Locals
    type(ParticlesFileType) :: tPart
    character(len=256)      :: sInputFile
    integer                 :: iRetCode
    integer                 :: iTimeStep
    integer                 :: iYear, iMonth, iDay, iHour, iMinute, iSecond
    
    ! Get parameters
    if(command_argument_count() /= 1) then
        print *, "dia_part - Procedure, for performing plausibility checks on particles file"
        print *
        print *, "Usage:"
        print *
        print *, "  ./dia_part <Particles_File>"
        print *
        print *, "Copyright 2020 by Servizi Territorio srl"
        print *, "                  This is open-source code, covered by the MIT license"
        print *
        stop
    end if
    call get_command_argument(1, sInputFile)
    print *, trim(sInputFile)
    
    ! Access input file
    iRetCode = tPart % connect(sInputFile)
    if(iRetCode /= 0) then
        print *, "dia_part:: error: Input file not opened"
        stop
    end if
    
    ! Main loop: Iterate over time steps
    do iTimeStep = 1, tPart % iNumTimeSteps
        iRetCode = tPart % get()
        if(iRetCode /= 0) cycle
        call UnpackTime(tPart % iCurTime, iYear, iMonth, iDay, iHour, iMinute, iSecond)
        write(6,"(i6,1x,i4.4,2('-',i2.2),'T',i2.2,2(':',i2.2),1x,i8,4(1x,f9.2))") &
            iTimeStep, &
            iYear, iMonth, iDay, iHour, iMinute, iSecond, &
            size(tPart % ivTimeStamp), &
            minval(tPart % rvX), maxval(tPart % rvX), &
            minval(tPart % rvY), maxval(tPart % rvY)
    end do
    
    ! Leave
    iRetCode = tPart % disconnect()
    
end program dia_part
