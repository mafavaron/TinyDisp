module Meteo

	use calendar

	implicit none
	
	private
	
	! Public interface
	public	:: MeteoType
	
	! Data type
	type MeteoType
		integer, dimension(:), allocatable	:: ivTimeStamp
		real, dimension(:), allocatable		:: rvU
		real, dimension(:), allocatable		:: rvV
		real, dimension(:), allocatable		:: rvW
		real, dimension(:), allocatable		:: rvStdDevU
		real, dimension(:), allocatable		:: rvStdDevV
		real, dimension(:), allocatable		:: rvStdDevW
		real, dimension(:), allocatable		:: rvCovUV
		real, dimension(:), allocatable		:: rvCovUW
		real, dimension(:), allocatable		:: rvCovVW
	contains
		procedure 							:: read		=> met_read
		procedure							:: resample	=> met_resample
	end type MeteoType

contains

	function met_read(this, iLUN, sFileName) result(iRetCode)
	
		! Routine arguments
		class(MeteoType), intent(out)	:: this
		integer, intent(in)				:: iLUN
		character(len=*), intent(in)	:: sFileName
		integer							:: iRetCode
		
		! Locals
		integer				:: iErrCode
		integer				:: iNumData
		integer				:: iData
		character(len=256)	:: sBuffer
		integer				:: iYear, iMonth, iDay, iHour, iMinute, iSecond
		logical				:: lSorted
		
		! Assume success (will falsify on failure)
		iRetCode = 0
		
		! Get data
		open(iLUN, file=sFileName, status='old', action='read', iostat=iErrCode)
		if(iErrCode /= 0) then
			iRetCode = 1
			return
		end if
		iNumData = -1	! Not 0, to account for the header line
		do
			read(iLUN, "(a)", iostat=iErrCode) sBuffer
			if(iErrCode /= 0) exit
			iNumData = iNumData + 1
		end do
		if(iNumData < 2) then
			iRetCode = 2
			close(iLUN)
			return
		end if
		rewind(iLUN)
		read(iLUN, "(a)") sBuffer	! Skip header (now, the "normal way"
		do iData = 1, iNumData
			read(iLUN, "(a)") sBuffer
			read(sBuffer(1:19), "(i4,5()1x,i2)") iYear, iMonth, iDay, iHour, iMinute, iSecond
			call PackTime(this % ivTimeStamp(iData), iYear, iMonth, iDay, iHour, iMinute, iSecond)
			read(21:), *) &
				this % rvU(iData), &
				this % rvV(iData), &
				this % rvW(iData), &
				this % rvStdDevU(iData), &
				this % rvStdDevV(iData), &
				this % rvStdDevW(iData), &
		end do
		close(iLUN)
		
		! Check data are sorted with respect to time
		lSorted = .TRUE.
		do iData = 2, iNumData
			if(this % ivTimeStamp(iData-1) >= this % ivTimeStamp(iData)) then
				lSorted = .FALSE.
				exit
			end if
		end do
		if(.not.lSorted) then
			iRetCode = 3
		end if
		
	end function met_read
	
	
	function met_resample(this, iTimeStep) result(iRetCode)

		! Routine arguments
		class(MeteoType), intent(out)	:: this
		integer, intent(in)				:: iTimeStep
		integer							:: iRetCode
		
		! Locals
		integer								:: iNumMeteoData
		integer, dimension(:), allocatable	:: ivTimeStamp
		real, dimension(:), allocatable		:: rvU
		real, dimension(:), allocatable		:: rvV
		real, dimension(:), allocatable		:: rvW
		real, dimension(:), allocatable		:: rvStdDevU
		real, dimension(:), allocatable		:: rvStdDevV
		real, dimension(:), allocatable		:: rvStdDevW
		real, dimension(:), allocatable		:: rvCovUV
		real, dimension(:), allocatable		:: rvCovUW
		real, dimension(:), allocatable		:: rvCovVW
		integer								:: iIdx
		integer								:: iNext
		integer								:: iTimeStamp
		integer								:: iLastTime
		integer								:: iNumElements
		
		! Assume success (will falsify on failure)
		iRetCode = 0
		
		! Reserve workspace
		iIdx         = 1
		iTimeStamp   = this % ivTimeStamp(iIdx)
		iLastTime    = this % ivTimeStamp(size(this % ivTimeStamp))
		iNumElements = (iLastTime - iTimeStamp) / iTimeStep
		if(iNumElements <= 1) then
			iRetCode = 1
			return
		end if
		allocate(ivTimeStamp(iNumElements))
		allocate(rvU(iNumElements))
		allocate(rvV(iNumElements))
		allocate(rvW(iNumElements))
		allocate(rvStdDevU(iNumElements))
		allocate(rvStdDevV(iNumElements))
		allocate(rvStdDevW(iNumElements))
		allocate(rvCovUV(iNumElements))
		allocate(rvCovUW(iNumElements))
		allocate(rvCovVW(iNumElements))
		
		! Main loop: locate sampling time stamps, and linearly interpolate the original data
		! at them
		iNext = 1
		do while iTimeStamp <= iLastTime

			if(iTimeStamp == ivTimeStamp(iIdx)) then

				! Exact time match: retrieve actual values
				ivTimeStamp(iNext) = iTimeStamp
				rvU(iNext)         = this % rvU(iIdx)
				rvV(iNext)         = this % rvV(iIdx)
				rvW(iNext)         = this % rvW(iIdx)
				rvStdDevU(iNext)   = this % rvStdDevU(iIdx)
				rvStdDevV(iNext)   = this % rvStdDevV(iIdx)
				rvStdDevW(iNext)   = this % rvStdDevW(iIdx)
				rvCovUV(iNext)     = this % rvCovUV(iIdx)
				rvCovUW(iNext)     = this % rvCovUW(iIdx)
				rvCovVW(iNext)     = this % rvCovVW(iIdx)

			else

				! Locate the first useful 'iIdx' value so that ivTimeStamp(iIdx) <= iTimeStamp < ivTimeStamp(iIdx+1)
				do while (iTimeStamp < iLastTime .and. iTimeStamp >= ivTimeStamp(iIdx + 1))
					iIdx = iIdx + 1
				end do

				if(iTimeStamp == ivTimeStamp(iIdx)) then
						
					! Exact time match: retrieve actual values
					ivTimeStamp(iNext) = iTimeStamp
					rvU(iNext)         = this % rvU(iIdx)
					rvV(iNext)         = this % rvV(iIdx)
					rvW(iNext)         = this % rvW(iIdx)
					rvStdDevU(iNext)   = this % rvStdDevU(iIdx)
					rvStdDevV(iNext)   = this % rvStdDevV(iIdx)
					rvStdDevW(iNext)   = this % rvStdDevW(iIdx)
					rvCovUV(iNext)     = this % rvCovUV(iIdx)
					rvCovUW(iNext)     = this % rvCovUW(iIdx)
					rvCovVW(iNext)     = this % rvCovVW(iIdx)

				else

					! Time is somewhere in-between: linear interpolation
						this->ivTimeStamp.push_back(iTimeStamp);
						float rFraction = (float)(iTimeStamp - ivTimeStamp[iIdx]) / (ivTimeStamp[iIdx + 1] - ivTimeStamp[iIdx]);
						this->rvU.push_back(rvU[iIdx] + rFraction * (rvU[iIdx + 1] - rvU[iIdx]));
						this->rvV.push_back(rvV[iIdx] + rFraction * (rvV[iIdx + 1] - rvV[iIdx]));
						this->rvStdDevU.push_back(rvStdDevU[iIdx] + rFraction * (rvStdDevU[iIdx + 1] - rvStdDevU[iIdx]));
						this->rvStdDevV.push_back(rvStdDevV[iIdx] + rFraction * (rvStdDevV[iIdx + 1] - rvStdDevV[iIdx]));
						this->rvCovUV.push_back(rvCovUV[iIdx] + rFraction * (rvCovUV[iIdx + 1] - rvCovUV[iIdx]));

				end if

			end if

			iTimeStamp = iTimeStamp + iTimeStep
				
		end do
		
		! Leave
		deallocate(rvCovVW)
		deallocate(rvCovUW)
		deallocate(rvCovUV)
		deallocate(rvStdDevW)
		deallocate(rvStdDevV)
		deallocate(rvStdDevU)
		deallocate(rvW)
		deallocate(rvV)
		deallocate(rvU)
		deallocate(ivTimeStamp)

	end function met_resample

end module Meteo
