! Test driver for module "Meteo.f90".
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program Meteo_Test

	use Meteo
	use Calendar
	
	implicit none
	
	! Locals
	type(MeteoType)	:: tMet
	integer			:: iRetCode
	integer			:: i
	integer			:: iYear, iMonth, iDay, iHour, iMinute, iSecond
	
	! Test 1: check read and interpolation under ideal conditions
	open(10, file='test.csv', status='unknown', action='write')
	write(10, "('Time.Stamp, U, V, W, StdDev.U, StdDev.V, StdDev.W, Cov.UV, Cov.UW, Cov.VW')")
	write(10, "('2020-04-01 00:00:00, 0., 1., 2., -1., -2., -3., 0.,  1.,  10.')")
	write(10, "('2020-04-02 00:00:00, 1., 2., 4., -2., -4., -9., 0., -1., -10.')")
	close(10)
	print *, 'Test 1 - Interpolation under ideal conditions'
	iRetCode = tMet % read(10, 'test.csv')
	if(iRetCode /= 0) then
		print *, 'Test 1 not passed - Reading meteo data - Return code = ', iRetCode
		stop
	end if
	iRetCode = tMet % resample(3600)
	if(iRetCode /= 0) then
		print *, 'Test 1 not passed - Resampling meteo data - Return code = ', iRetCode
		stop
	end if
	open(10, file='.\\test1.out', status='unknown', action='write')
	write(10, "('Time.Stamp, U, V, W, StdDev.U, StdDev.V, StdDev.W, Cov.UV, Cov.UW, Cov.VW')")
	do i = 1, size(tMet % ivTimeStamp)
		call UnpackTime(tMet % ivTimeStamp(i), iYear, iMonth, iDay, iHour, iMinute, iSecond)
		write(10, "(i4.4,2('-',i2.2),'T',i2.2,2(':',i2.2),9(',',f7.3))") &
			iYear, iMonth, iDay, iHour, iMinute, iSecond, &
			tMet % rvU(i), &
			tMet % rvV(i), &
			tMet % rvW(i), &
			tMet % rvStdDevU(i), &
			tMet % rvStdDevV(i), &
			tMet % rvStdDevW(i), &
			tMet % rvCovUV(i), &
			tMet % rvCovUW(i), &
			tMet % rvCovVW(i)
	end do
	close(10)
	
	! Test 2: Check interpolation using a non-dividing time step
	open(10, file='test.csv', status='unknown', action='write')
	write(10, "('Time.Stamp, U, V, W, StdDev.U, StdDev.V, StdDev.W, Cov.UV, Cov.UW, Cov.VW')")
	write(10, "('2020-04-01 00:00:00, 0., 1., 2., -1., -2., -3., 0.,  1.,  10.')")
	write(10, "('2020-04-02 00:00:00, 1., 2., 4., -2., -4., -9., 0., -1., -10.')")
	close(10)
	print *, 'Test 2 - Interpolation using non-divisor'
	iRetCode = tMet % read(10, 'test.csv')
	if(iRetCode /= 0) then
		print *, 'Test 2 not passed - Reading meteo data - Return code = ', iRetCode
		stop
	end if
	iRetCode = tMet % resample(3601)
	if(iRetCode /= 0) then
		print *, 'Test 2 not passed - Resampling meteo data - Return code = ', iRetCode
		stop
	end if
	open(10, file='.\\test2.out', status='unknown', action='write')
	write(10, "('Time.Stamp, U, V, W, StdDev.U, StdDev.V, StdDev.W, Cov.UV, Cov.UW, Cov.VW')")
	do i = 1, size(tMet % ivTimeStamp)
		call UnpackTime(tMet % ivTimeStamp(i), iYear, iMonth, iDay, iHour, iMinute, iSecond)
		write(10, "(i4.4,2('-',i2.2),'T',i2.2,2(':',i2.2),9(',',f7.3))") &
			iYear, iMonth, iDay, iHour, iMinute, iSecond, &
			tMet % rvU(i), &
			tMet % rvV(i), &
			tMet % rvW(i), &
			tMet % rvStdDevU(i), &
			tMet % rvStdDevV(i), &
			tMet % rvStdDevW(i), &
			tMet % rvCovUV(i), &
			tMet % rvCovUW(i), &
			tMet % rvCovVW(i)
	end do
	close(10)
	
end program Meteo_Test

