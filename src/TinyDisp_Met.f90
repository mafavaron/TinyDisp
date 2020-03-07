! Main program: iterate over FastSonic files, get their sonic part,
!               and compute some diagnostic statistics on them. Write
!               a report from results.
!
! By: Patrizia M. Favaron
!
program TinyDisp_Met

	use files
	use stats_dia

	implicit none

	! Locals
	character(len=256)  :: sInputPath
	character(len=256)  :: sOutputFile
	integer             :: iRetCode
	real                :: rTimeBegin
	real                :: rTimeEnd
	real                :: rMaxSec
	integer             :: i
	integer             :: iData
	integer             :: iMinute, iSecond
	character(len=256), dimension(:), allocatable	:: svFiles
	real(4), dimension(:), allocatable				:: rvTimeStamp
	real(4), dimension(:), allocatable				:: rvU
	real(4), dimension(:), allocatable				:: rvV
	real(4), dimension(:), allocatable				:: rvW
	real(4), dimension(:), allocatable				:: rvT
	real(4), dimension(:), allocatable				:: rvAvgTime
	real(4), dimension(:), allocatable				:: rvAvgU
	real(4), dimension(:), allocatable				:: rvAvgV
	real(4), dimension(:), allocatable				:: rvStdDevU
	real(4), dimension(:), allocatable				:: rvStdDevV
	real(4), dimension(:), allocatable				:: rvCovUV
	real(4), dimension(:,:), allocatable			:: rmQuantity
	character(8), dimension(:), allocatable			:: svQuantity
	integer, dimension(1)							:: ivPos
	integer             :: iDateStart
	character(len=4)    :: sYear
	character(len=2)    :: sMonth
	character(len=2)    :: sDay
	character(len=2)    :: sHour
	character(len=32)   :: sDateTime

	! Get command arguments
	if(command_argument_count() /= 2) then
		print *, "tdm - Convert raw data to TinyDisp meteo form"
		print *
		print *, "Usage:"
		print *
		print *, "  ./tdm <Input_Path> <Output_File>"
		print *
		print *, "Copyright 2020 by Servizi Territorio srl"
		print *, "                  This software is open source, covered by the MIT license"
		print *
		stop
	end if
	call get_command_argument(1, sInputPath)
	call get_command_argument(2, sOutputFile)

	! Time elapsed counts
	call cpu_time(rTimeBegin)

	! Identify sub-directories in input path
	iRetCode = FindDataFiles(sInputPath, svFiles, PATH$FLAT, .FALSE.)
	if(iRetCode /= 0) then
		print *, 'error:: Invalid directory structure type'
		stop
	end if

	! Main loop: process files in turn
	open(11, file=sOutputFile, status='unknown', action='write')
	write(11, "('Time.stamp,U,V,SigmaU,SigmaV,CovUV')")
	do i = 1, size(svFiles)

		! Get date and hour from file name
		iDateStart = len_trim(svFiles(i)) - 14
		sYear  = svFiles(i)(iDateStart:iDateStart+3)
		sMonth = svFiles(i)(iDateStart+4:iDateStart+5)
		sDay   = svFiles(i)(iDateStart+6:iDateStart+7)
		sHour  = svFiles(i)(iDateStart+9:iDateStart+10)
		write(sDateTime, "(a4,2('-',a2),1x,a2,':00:00')") sYear, sMonth, sDay, sHour

		! Get file
		iRetCode = fsGet(svFiles(i), rvTimeStamp, rvU, rvV, rvW, rvT, rmQuantity, svQuantity)
		if(iRetCode /= 0) then
			print *, 'error:: File termination before completing data read'
			stop
		end if
		
		! Compute the hourly mean of U and V wind components (max of 3s moving average would be the WMO definition of wind gust)
		iRetCode = mean(rvTimeStamp, rvU, 600., rvAvgTime, rvAvgU)
		if(iRetCode /= 0) then
			print *, 'warning:: Some problem computing 600s mean'
			cycle
		end if
		iRetCode = mean(rvTimeStamp, rvV, 600., rvAvgTime, rvAvgV)
		if(iRetCode /= 0) then
			print *, 'warning:: Some problem computing 600s mean'
			cycle
		end if
		iRetCode = stddev(rvTimeStamp, rvU, 600., rvAvgTime, rvStdDevU)
		if(iRetCode /= 0) then
			print *, 'warning:: Some problem computing 600s mean'
			cycle
		end if
		iRetCode = stddev(rvTimeStamp, rvV, 600., rvAvgTime, rvStdDevV)
		if(iRetCode /= 0) then
			print *, 'warning:: Some problem computing 600s mean'
			cycle
		end if
		iRetCode = cov(rvTimeStamp, rvU, rvV, 600., rvAvgTime, rvCovUV)
		if(iRetCode /= 0) then
			print *, 'warning:: Some problem computing 600s mean'
			cycle
		end if

		! Print all data computed so far
		do iData = 1, size(rvAvgTime)
		
			iMinute = floor(rvAvgTime(iData) / 60.)
			iSecond = floor(rvAvgTime(iData) - iMinute * 60.)
		
			write(11, "(a4,2('-',a2),1x,a2,2(':',i2.2),5(',',f10.4))") &
				sYear, sMonth, sDay, sHour, iMinute, iSecond, &
				rvAvgU(iData), rvAvgV(iData), &
				rvStdDevU(iData), rvStdDevV(iData), &
				rvCovUV(iData)
				
		end do

	end do
	close(11)

	! Time elapsed counts
	call cpu_time(rTimeEnd)
	print *, "*** END JOB *** (Time elapsed:", rTimeEnd - rTimeBegin, ")"

end program TinyDisp_Met
