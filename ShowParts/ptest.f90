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
    end do

    ! Leave
    iRetCode = tPart % Close()
    
end program ptest
