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

end program met_split
