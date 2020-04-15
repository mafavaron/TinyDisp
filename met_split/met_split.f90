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
    character(len=256)                  :: sFileName
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
    real, dimension(:), allocatable     :: rvVel
    real, dimension(:), allocatable     :: rvUnitU
    real, dimension(:), allocatable     :: rvUnitV
    integer, dimension(:), allocatable  :: ivDayBegin
    integer, dimension(:), allocatable  :: ivDayEnd
    integer, dimension(:), allocatable  :: ivDayStamp
    integer                             :: iNumDays
    real, dimension(:), allocatable     :: rvminVel
    real, dimension(:), allocatable     :: rvMaxVel
    real, dimension(:), allocatable     :: rvVectorVel
    real, dimension(:), allocatable     :: rvScalarVel
    real, dimension(:), allocatable     :: rvCircularVar
    character(len=256)                  :: sHeader
    character(len=256)                  :: sBuffer
    integer                             :: iYear, iMonth, iDay, iHour, iMinute, iSecond
    integer                             :: iCurTime
    integer                             :: iOldDay
    integer                             :: iCurDay
    integer                             :: iDayIdx
    integer                             :: iBegin
    integer                             :: iEnd
    integer                             :: iDataInDay
    
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
    allocate(rvVel(iNumLines))
    allocate(rvUnitU(iNumLines))
    allocate(rvUnitV(iNumLines))
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
    iNumDays = iNumDays + 1
    allocate(ivDayBegin(iNumDays))
    allocate(ivDayEnd(iNumDays))
    allocate(ivDayStamp(iNumDays))
    
    ! Delimit days
    iOldDay = ivTimeStamp(1) / 86400
    iDayIdx = 0
    ivDayBegin(1) = 1
    ivDayStamp(1) = iOldDay * 86400
    do iLine = 2, iNumLines
        iCurTime = ivTimeStamp(iLine)
        iCurDay = iCurTime / 86400
        if(iCurDay /= iOldDay) then
            iDayIdx = iDayIdx + 1
            ivDayEnd(iDayIdx) = iLine - 1
            ivDayBegin(iDayIdx + 1) = iLine
            ivDayStamp(iDayIdx + 1) = iCurDay * 86400
            iOldDay = iCurDay
        end if
    end do
    iDayIdx = iDayIdx + 1
    ivDayEnd(iDayIdx) = iNumLines
    
    ! Compute the descriptive statistics
    rvVel = sqrt(rvU**2 + rvV**2)
    where(rvVel > 0.)
        rvUnitU = rvU / rvVel
        rvUnitV = rvV / rvVel
    elsewhere
        rvUnitU = 0.
        rvUnitV = 0.
    endwhere
    allocate(rvMinVel(iNumDays))
    allocate(rvMaxVel(iNumDays))
    allocate(rvVectorVel(iNumDays))
    allocate(rvScalarVel(iNumDays))
    allocate(rvCircularVar(iNumDays))
    do iDayIdx = 1, iNumDays
        iBegin = ivDayBegin(iDayIdx)
        iEnd   = ivDayEnd(iDayIdx)
        iDataInDay = iEnd - iBegin + 1
        rvMinVel(iDayIdx) = minval(rvVel(iBegin:iEnd))
        rvMaxVel(iDayIdx) = maxval(rvVel(iBegin:iEnd))
        rvScalarVel(iDayIdx) = sum(rvVel(iBegin:iEnd)) / iDataInDay
        rvVectorVel(iDayIdx) = sqrt( &
            (sum(rvU(iBegin:iEnd)) / iDataInDay)**2 + &
            (sum(rvV(iBegin:iEnd)) / iDataInDay)**2   &
        )
        rvCircularVar(iDayIdx) = 1. - sqrt( &
            (sum(rvUnitU(iBegin:iEnd)) / iDataInDay)**2 + &
            (sum(rvUnitV(iBegin:iEnd)) / iDataInDay)**2   &
        )
    end do
    
    ! Write diagnostic data
    open(10, file=sDiaFile, status='unknown', action='write')
    write(10, "('Time.Stamp, Min.Vel, Max.Vel, Vector.Vel, Scalar.Vel, Circ.Var')")
    do iDayIdx = 1, iNumDays
        call unpacktime(ivDayStamp(iDayIdx), iYear, iMonth, iDay, iHour, iMinute, iSecond)
        write(10, "(i4.4,2('-',i2.2),1x,i2.2,2(':',i2.2),4(',',f6.3),',',f6.4)") &
            iYear, iMonth, iDay, iHour, iMinute, iSecond, &
            rvMinVel(iDayIdx), rvMaxVel(iDayIdx), &
            rvVectorVel(iDayIdx), rvScalarVel(iDayIdx), &
            rvCircularVar(iDayIdx)
    end do
    close(10)
    
    ! Divide the original file in daily blocks
    do iDayIdx = 1, iNumDays
        iBegin = ivDayBegin(iDayIdx)
        iEnd   = ivDayEnd(iDayIdx)
        call unpacktime(ivDayStamp(iDayIdx), iYear, iMonth, iDay, iHour, iMinute, iSecond)
        write(sFileName, "(a,i4.4,2(i2.2),'.csv')") &
            trim(sOutputPrefix), &
            iYear, iMonth, iDay
        open(10, file=sFileName, status='unknown', action='write')
        write(10, "(a)") sHeader
        do iLine = iBegin, iEnd
            call unpacktime(ivTimeStamp(iLine), iYear, iMonth, iDay, iHour, iMinute, iSecond)
            write(10, "(i4.4,2('-',i2.2),1x,i2.2,2(':',i2.2),3(',',f8.2),6(',',f8.4))") &
                iYear, iMonth, iDay, iHour, iMinute, iSecond, &
                rvU(iLine), rvV(iLine), rvW(iLine), &
                rvStdDevU(iLine), rvStdDevV(iLine), rvStdDevW(iLine), &
                rvCovUV(iLine), rvCovUW(iLine), rvCovVW(iLine)
        end do
        close(10)
    end do
    
    ! Leave
    deallocate(rvCircularVar)
    deallocate(rvScalarVel)
    deallocate(rvVectorVel)
    deallocate(rvMaxVel)
    deallocate(rvMinVel)
    deallocate(ivDayEnd)
    deallocate(ivDayBegin)
    deallocate(rvUnitV)
    deallocate(rvUnitU)
    deallocate(rvVel)
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
