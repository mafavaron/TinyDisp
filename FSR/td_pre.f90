! Main program of td_pre, the "preparer" of meteorological data for TinyDisp.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program td_pre

    use fileList
    use files
    use stats_dia

    implicit none
    
    ! Locals
    character(len=256)                              :: sFsrList
    character(len=16)                               :: sBuffer
    integer                                         :: iAvgTime
    character(len=256)                              :: sOutFile
    integer                                         :: iRetCode
    character(len=256), dimension(:), allocatable   :: svFiles
    integer                                         :: i, j
    integer                                         :: iDateStart
    character(len=4)                                :: sYear
    character(len=2)                                :: sMonth
    character(len=2)                                :: sDay
    character(len=2)                                :: sHour
    integer                                         :: iMinute
    integer                                         :: iSecond
    character(len=32)                               :: sDateTime
    real(4), dimension(:), allocatable              :: rvTimeStamp
    real(4), dimension(:), allocatable              :: rvU
    real(4), dimension(:), allocatable              :: rvV
    real(4), dimension(:), allocatable              :: rvW
    real(4), dimension(:), allocatable              :: rvT
    real(4), dimension(:,:), allocatable            :: rmQuantity
    character(8), dimension(:), allocatable         :: svQuantity
    real(4), dimension(:), allocatable              :: rvVel
    real(4), dimension(:), allocatable              :: rvAvgTime
    real(4), dimension(:), allocatable              :: rvAvgVel
    real(4), dimension(:), allocatable              :: rvAvgU
    real(4), dimension(:), allocatable              :: rvAvgV
    real(4), dimension(:), allocatable              :: rvAvgW
    real(4), dimension(:), allocatable              :: rvAvgT
    real(4), dimension(:), allocatable              :: rvStdDevU
    real(4), dimension(:), allocatable              :: rvStdDevV
    real(4), dimension(:), allocatable              :: rvStdDevW
    real(4), dimension(:), allocatable              :: rvStdDevT
    real(4), dimension(:), allocatable              :: rvCovUV
    real(4), dimension(:), allocatable              :: rvCovUW
    real(4), dimension(:), allocatable              :: rvCovVW
    real(4), dimension(:), allocatable              :: rvCovUT
    real(4), dimension(:), allocatable              :: rvCovVT
    real(4), dimension(:), allocatable              :: rvCovWT
    
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
        stop
    end if
    call get_command_argument(1, sFsrList)
    call get_command_argument(2, sBuffer)
    read(sBuffer, *, iostat=iRetCode) iAvgTime
    if(iRetCode /= 0) then
        print *, 'td_pre:: error: Invalid averaging time.'
        stop
    end if
    call get_command_argument(3, sOutFile)
    
    ! Read file list
    iRetCode = readFileList(sFsrList, svFiles)
    if(iRetCode /= 0) then
        print *, 'td_pre:: error: Data files list not read.'
        stop
    end if

    ! Main loop: process files in turn
    open(10, file=sOutFile, status='unknown', action='write')
    write(10, "('Time.Stamp, U, V, W, StdDev.U, StdDev.V, StdDev.W, Cov.UV, Cov.UW, Cov.VW')")
    do i = 1, size(svFiles)

        ! Get date and hour from file name
        iDateStart = len_trim(svFiles(i)) - 15
        sYear  = svFiles(i)(iDateStart:iDateStart+3)
        sMonth = svFiles(i)(iDateStart+4:iDateStart+5)
        sDay   = svFiles(i)(iDateStart+6:iDateStart+7)
        sHour  = svFiles(i)(iDateStart+9:iDateStart+10)
        write(sDateTime, "(a4,2('-',a2),1x,a2,':00:00')") sYear, sMonth, sDay, sHour

		! Get file
        iRetCode = fileRead(svFiles(i), rvTimeStamp, rvU, rvV, rvW, rvT, rmQuantity, svQuantity)
        if(iRetCode /= 0) then
            print *, 'td_pre:: error: File termination before completing data read'
            stop
        end if
		
        ! Combine horizontal components into a unique wind speed
        if(allocated(rvVel)) deallocate(rvVel)
        allocate(rvVel(size(rvU)))
        rvVel = sqrt(rvU**2 + rvV**2)
		
        ! Compute wind and temperature statistics
        iRetCode = mean(rvTimeStamp, rvVel, real(iAvgTime), rvAvgTime, rvAvgVel)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = mean(rvTimeStamp, rvU, real(iAvgTime), rvAvgTime, rvAvgU)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = mean(rvTimeStamp, rvV, real(iAvgTime), rvAvgTime, rvAvgV)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = mean(rvTimeStamp, rvW, real(iAvgTime), rvAvgTime, rvAvgW)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = mean(rvTimeStamp, rvT, real(iAvgTime), rvAvgTime, rvAvgT)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = stddev(rvTimeStamp, rvU, real(iAvgTime), rvAvgTime, rvStdDevU)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = stddev(rvTimeStamp, rvV, real(iAvgTime), rvAvgTime, rvStdDevV)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = stddev(rvTimeStamp, rvW, real(iAvgTime), rvAvgTime, rvStdDevW)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = stddev(rvTimeStamp, rvT, real(iAvgTime), rvAvgTime, rvStdDevT)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = cov(rvTimeStamp, rvU, rvV, real(iAvgTime), rvAvgTime, rvCovUV)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = cov(rvTimeStamp, rvU, rvW, real(iAvgTime), rvAvgTime, rvCovUW)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = cov(rvTimeStamp, rvV, rvW, real(iAvgTime), rvAvgTime, rvCovVW)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = cov(rvTimeStamp, rvU, rvT, real(iAvgTime), rvAvgTime, rvCovUT)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = cov(rvTimeStamp, rvV, rvT, real(iAvgTime), rvAvgTime, rvCovVT)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
        iRetCode = cov(rvTimeStamp, rvW, rvT, real(iAvgTime), rvAvgTime, rvCovWT)
        if(iRetCode /= 0) then
            print *, 'td_pre:: warning: Some problem computing mean'
            cycle
        end if
		
		! Write data to TinyDisp file
        do j = 1, size(rvAvgTime)
            iSecond = floor(rvAvgTime(j))
            iMinute = (iSecond / 60)
            iSecond = iSecond - iMinute * 60
            write(10, "(a4,2('-',a2),1x,a2,2(':',i2.2),9(',',f8.2))") &
                sYear, sMonth, sDay, sHour, iMinute, iSecond, &
                rvAvgU(j), rvAvgV(j), rvAvgW(j), &
                rvStdDevU(j), rvStdDevV(j), rvStdDevW(j), &
                rvCovUV(j), rvCovUW(j), rvCovVW(j)
        end do

    end do
    close(10)

    ! Leave
    print *, "*** END JOB ***"

end program td_pre
