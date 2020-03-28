module Meteo

	implicit none
	
	private
	
	! Public interface
	
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
		rewind(iLUN)
		read(iLUN, "(a)") sBuffer	! Skip header (now, the "normal way"
		do iData = 1, iNumData
			read(iLUN, "(a)") sBuffer
			read(sBuffer(1:19), "(i4,5()1x,i2)") iYear, iMonth, iDay, iHour, iMinute, iSecond
			read(21:), *) &
				this % rvU(iData), &
				this % rvV(iData), &
				this % rvW(iData), &
				this % rvStdDevU(iData), &
				this % rvStdDevV(iData), &
				this % rvStdDevW(iData), &
		end do
		close(iLUN)
		
	end function met_read
	
	
	function met_resample(this) result(iRetCode)

		! Routine arguments
		class(MeteoType), intent(out)	:: this
		integer							:: iRetCode
		
			// Gather meteo data. By construction, meteo data are sorted
			// ascending with respect to time stamps
			std::string			sBuffer;
			bool				lIsFirst = true;
			std::vector<time_t>	ivTimeStamp;
			std::vector<float>	rvU;
			std::vector<float>	rvV;
			std::vector<float>	rvStdDevU;
			std::vector<float>	rvStdDevV;
			std::vector<float>	rvCovUV;
			while (std::getline(fMeteo, sBuffer)) {
				if (lIsFirst) {
					lIsFirst = false; // And, do nothing with the buffer - a header, in case
				}
				else {
					static const std::string dateTimeFormat{ "%Y-%m-%d %H:%M:%S" };
					std::vector<std::string> svFields;
					split(sBuffer, svFields);
					if (svFields.size() == 6) {
						float rU       = stof(svFields[1]);
						float rV       = stof(svFields[2]);
						float rStdDevU = stof(svFields[3]);
						float rStdDevV = stof(svFields[4]);
						float rCovUV   = stof(svFields[5]);
						if (rU > -9999.0f && rV > -9999.0f && rStdDevU > -9999.0f && rStdDevV > -9999.0f && rCovUV > -9999.0f) {
							std::istringstream ss{svFields[0]};
							struct tm tTimeStamp;
							ss >> std::get_time(&tTimeStamp, dateTimeFormat.c_str());
							ivTimeStamp.push_back(std::mktime(&tTimeStamp));
							rvU.push_back(rU);
							rvV.push_back(rV);
							rvStdDevU.push_back(rStdDevU);
							rvStdDevV.push_back(rStdDevV);
							rvCovUV.push_back(rCovUV);
						}
					}
				}
			}

			// Check some meteo data has been collected (if not, there is
			// nothing to be made
			if (ivTimeStamp.size() <= 0) return;

			// Check configuration values
			if (this->iTimeStep <= 0) return;
			if (this->rEdgeLength <= 0.0f) return;
			if (this->iPartsPerStep <= 0) return;
			if (this->iStepsSurvival <= 0) return;

			// Interpolate linearly in the time range of meteo data
			std::vector<float> rvInterpDeltaTime;
			int				   iIdx = 0;
			time_t             iTimeStamp = ivTimeStamp[iIdx];
			time_t             iLastTime = ivTimeStamp[ivTimeStamp.size() - 1];
			int                iNumElements = (iLastTime - iTimeStamp) / this->iTimeStep;
			this->ivTimeStamp.reserve(iNumElements);
			this->rvU.reserve(iNumElements);
			this->rvV.reserve(iNumElements);
			this->rvStdDevU.reserve(iNumElements);
			this->rvStdDevV.reserve(iNumElements);
			this->rvCovUV.reserve(iNumElements);
			while (iTimeStamp <= iLastTime) {

				// Exactly the same?
				if (iTimeStamp == ivTimeStamp[iIdx]) {

					// Yes! Just get values
					this->ivTimeStamp.push_back( iTimeStamp);
					this->rvU.push_back(         rvU[iIdx]       );
					this->rvV.push_back(         rvV[iIdx]       );
					this->rvStdDevU.push_back(   rvStdDevU[iIdx] );
					this->rvStdDevV.push_back(   rvStdDevV[iIdx] );
					this->rvCovUV.push_back(     rvCovUV[iIdx]   );

				}
				else {

					// No: Locate iIdx so that ivTimeStamp[iIdx] <= iTimeStamp < ivTimeStamp[iIdx+1]
					while (iTimeStamp < iLastTime && iTimeStamp >= ivTimeStamp[iIdx + 1]) {
						++iIdx;
					}

					// Check whether time is the same or not
					if (iTimeStamp == ivTimeStamp[iIdx]) {
						
						// Same! Just get values
						this->ivTimeStamp.push_back(iTimeStamp);
						this->rvU.push_back(rvU[iIdx]);
						this->rvV.push_back(rvV[iIdx]);
						this->rvStdDevU.push_back(rvStdDevU[iIdx]);
						this->rvStdDevV.push_back(rvStdDevV[iIdx]);
						this->rvCovUV.push_back(rvCovUV[iIdx]);

					}
					else {

						// Somewhere in-between: linear interpolation
						this->ivTimeStamp.push_back(iTimeStamp);
						float rFraction = (float)(iTimeStamp - ivTimeStamp[iIdx]) / (ivTimeStamp[iIdx + 1] - ivTimeStamp[iIdx]);
						this->rvU.push_back(rvU[iIdx] + rFraction * (rvU[iIdx + 1] - rvU[iIdx]));
						this->rvV.push_back(rvV[iIdx] + rFraction * (rvV[iIdx + 1] - rvV[iIdx]));
						this->rvStdDevU.push_back(rvStdDevU[iIdx] + rFraction * (rvStdDevU[iIdx + 1] - rvStdDevU[iIdx]));
						this->rvStdDevV.push_back(rvStdDevV[iIdx] + rFraction * (rvStdDevV[iIdx + 1] - rvStdDevV[iIdx]));
						this->rvCovUV.push_back(rvCovUV[iIdx] + rFraction * (rvCovUV[iIdx + 1] - rvCovUV[iIdx]));

					}

				}

				iTimeStamp += this->iTimeStep;

			}

end module Meteo
