! Main program of td_pre, the "preparer" of meteorological data for TinyDisp.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program td_pre

    implicit none
    
    ! Locals
    character(len=256)  :: sFsrList
    character(len=256)  :: sFileName
    character(len=16)   :: sBuffer
    integer             :: iAvgTime
    character(len=256)  :: sOutFile
    integer             :: iRetCode
    
    ! Get command arguments
    if(command_argument_count() /= 3) then
        print *, 'td_pre - Program to translate a set of FSR ultrasonic anemometer raw data files'
        print *, '         to a TinyDisp wind-and-turbulence input file'
        print *
        print *, 'Usage:'
        print *
        print *, '  td_pre <FSR file list> <Averaging time> <Output file>'
        print *
        print *, 'The <Averaging time> is a positive integer number, expressed in seconds.'
        print *
        print *, 'Copyright 2020 by Servizi Territorio srl'
        print *, '                  This is open-source software, covered by the MIT license.'
        print *
        stop 1
    end if
    call get_command_argument(1, sFsrList)
    call get_command_argument(2, sBuffer)
    read(sBuffer, *, iostat=iRetCode) iAvgTime
    if(iRetCode /= 0) then
        print *, 'td_pre:: error: Invalid averaging time.'
        stop 2
    end if
    call get_command_argument(3, sOutFile)

end program td_pre
