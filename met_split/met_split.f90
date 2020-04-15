! Main program of met_split, the analyzer and day-splitter of
! meteorological data for TinyDisp.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program met_split

    use calendar
    
    implicit none
    
    ! Local variables
    integer                             :: iRetCode
    character(len=256)                  :: sInputFile
    character(len=256)                  :: sDiaFile
    character(len=256)                  :: sOutputPrefix
    integer, dimension(:), allocatable  :: ivTimeStamp
    real, dimension(:), allocatable     :: rvU
    real, dimension(:), allocatable     :: rvV
    real, dimension(:), allocatable     :: rvW
    real, dimension(:), allocatable     :: rvStdDevU
    real, dimension(:), allocatable     :: rvStdDevV
    real, dimension(:), allocatable     :: rvStdDevW
    real, dimension(:), allocatable     :: rvCovUV
    real, dimension(:), allocatable     :: rvCovUW
    real, dimension(:), allocatable     :: rvCovVW
    integer, dimension(:), allocatable  :: ivDayBegin
    integer, dimension(:), allocatable  :: ivDayEnd
    integer                             :: iNumDays
    character(len=256)                  :: sBuffer
    
    ! Get command arguments
    if(command_argument_count() /= 3) then
        print *, 'met_split - Program to analyze and day-split a TinyDisp meteo file'
        print *
        print *, 'Usage:'
        print *
        print *, '  met_split <Met file> <Dia file> <Output prefix>'
        print *
        print *, 'Copyright 2020 by Servizi Territorio srl'
        print *, '                  This is open-source software, covered by the MIT license.'
        print *
        stop
    end if
    call get_command_argument(1, sInputFile)
    call get_command_argument(2, sDiaFile)
    call get_command_argument(3, sOutputPrefix)
    
    ! Read meteo file
    open(10, file=sInputFile, status='old', action='read', iostat=iRetCode)
    close(10)
    
    ! Leave
    print *, '*** END JOB ***'

end program met_split
