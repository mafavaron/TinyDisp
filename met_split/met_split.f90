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
    integer                             :: iNumLines
    integer                             :: iLine
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
    character(len=256)                  :: sHeader
    character(len=256)                  :: sBuffer
    integer                             :: iYear, iMonth, iDay, iHour, iMinute, iSecond
    integer                             :: iCurTime
    integer                             :: iOldDay
    integer                             :: iCurDay
    integer                             :: iDayIdx
    
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
    iNumLines = 0
    open(10, file=sInputFile, status='old', action='read', iostat=iRetCode)
    if(iRetCode /= 0) then
        print *, 'met_split:: error: Meteorological input file not opened'
        stop
    end if
    read(10, "(a)", iostat=iRetCode) sHeader
    if(iRetCode /= 0) then
        print *, 'met_split:: error: Meteorological input file is empty'
        stop
    end if
    do
        read(10, "(a)", iostat=iRetCode) sBuffer
        if(iRetCode /= 0) exit
        iNumLines = iNumLines + 1
    end do
    rewind(10)
    if(iNumLines <= 1) then
        print *, 'met_split:: error: Meteorological input file is empty or contains a single data line'
        stop
    end if
    allocate(ivTimeStamp(iNumLines))
    allocate(rvU(iNumLines))
    allocate(rvV(iNumLines))
    allocate(rvW(iNumLines))
    allocate(rvStdDevU(iNumLines))
    allocate(rvStdDevV(iNumLines))
    allocate(rvStdDevW(iNumLines))
    allocate(rvCovUV(iNumLines))
    allocate(rvCovUW(iNumLines))
    allocate(rvCovVW(iNumLines))
    read(10, "(a)") sHeader
    do iLine = 1, iNumLines
        read(10, "(a)") sBuffer
        read(sBuffer, "(i4,5(1x,i2))") iYear, iMonth, iDay, iHour, iMinute, iSecond
        call packtime(ivTimeStamp(iLine), iYear, iMonth, iDay, iHour, iMinute, iSecond)
        read(sBuffer(21:),*) &
            rvU(iLine), &
            rvV(iLine), &
            rvW(iLine), &
            rvStdDevU(iLine), &
            rvStdDevV(iLine), &
            rvStdDevW(iLine), &
            rvCovUV(iLine), &
            rvCovUW(iLine), &
            rvCovVW(iLine)
    end do
    close(10)
    
    ! Count number of days in meteo file
    iOldDay = ivTimeStamp(1) / 86400
    iNumDays = 0
    do iLine = 2, iNumLines
        iCurTime = ivTimeStamp(iLine)
        iCurDay = iCurTime / 86400
        if(iCurDay /= iOldDay) then
            iNumDays = iNumDays + 1
            iOldDay = iCurDay
        end if
    end do
    allocate(ivDayBegin(iNumDays))
    allocate(ivDayEnd(iNumDays))
    
    ! Delimit days
    iOldDay = ivTimeStamp(1) / 86400
    iDayIdx = 0
    ivDayBegin(1) = 1
    do iLine = 2, iNumLines
        iCurTime = ivTimeStamp(iLine)
        iCurDay = iCurTime / 86400
        if(iCurDay /= iOldDay) then
            iDayIdx = iDayIdx + 1
            ivDayEnd(iDayIdx) = iLine - 1
            ivDayBegin(iDayIdx + 1) = iLine
            iOldDay = iCurDay
        end if
    end do
    ivDayEnd(iNumDays) = iNumLines
    
    do iDayIdx = 1, iNumDays
        print *, iDayIdx, ivDayBegin(iDayIdx), ivDayEnd(iDayIdx), ivDayEnd(iDayIdx)-ivDayBegin(iDayIdx)+1
    end do
    
    ! Leave
    deallocate(ivDayEnd)
    deallocate(ivDayBegin)
    deallocate(rvCovVW)
    deallocate(rvCovUW)
    deallocate(rvCovUV)
    deallocate(rvStdDevW)
    deallocate(rvStdDevV)
    deallocate(rvStdDevU)
    deallocate(rvW)
    deallocate(rvV)
    deallocate(rvU)
    print *, '*** END JOB ***'

end program met_split
