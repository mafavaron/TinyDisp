! Main program of dia_part
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program dia_part

    use TinyDispFiles
    
    implicit none
    
    ! Locals
    type(ParticlesFileType) :: tPart
    character(len=256)      :: sInputFile
    integer                 :: iRetCode
    integer                 :: iTimeStep
    
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
        print *, iTimeStep, tPart % iCurTime, size(tPart % ivTimeStamp)
    end do
    
    ! Leave
    iRetCode = tPart % disconnect()
    
end program dia_part
