module Processing

	use pbl_met
	use Meteo

	implicit none

	type Config
		! Status
		logical							:: lIsFull = .false.
		! General
		integer							:: iDebugLevel
		character(len=256)	:: sDiaFile
		integer							:: iFrameInterval
		character(len=256)	:: sFramePath
		integer							:: iExecMode
		! Timing
		integer							:: iAvgTime
		integer							:: iNumStep
		integer							:: iNumPart
		integer							:: iMaxAge
		! Emission
		character(len=256)	:: sStatic
		character(len=256)	:: sDynamic
		! Meteo
		character(len=256)	:: sMetInpFile
		character(len=256)	:: sMetOutFile
		character(len=256)	:: sMetDiaFile
		real(8)							:: rHeight
		real(8)							:: rZ0
		real(8)							:: rZr
		real(8)							:: rZt
		real(8)							:: rGamma
		integer							:: iHemisphere	! 0:Southern, 1:Northern
		! Output
		character(len=256)	:: sConcFile
		real(8)							:: rX0
		real(8)							:: rY0
		integer							:: iNx
		integer							:: iNy
		integer							:: iNz
		real(8)							:: rDx
		real(8)							:: rDy
		real(8)							:: rDz
		! Site parameters of meteorological file
		! Computed parameters
		real(8)							:: x1
		real(8)							:: y1
		real(8)							:: zmax
		! Meteo data
		type(MetData)				:: tMeteo
	contains
		procedure						:: read               => cfgRead
		procedure						:: getNumTimeSteps    => cfgGetTimeSteps
		procedure						:: getNumTimeSubSteps => cfgGetTimeSubSteps
		procedure						:: getNumMeteo        => cfgGetMeteoSize
	end type Config


	type MetProfiles
		! Time stamp
		real(8)								:: rEpoch	! Time stamp of current profile set
		! Primitive profiles
		real(8), dimension(:), allocatable	:: z		! Levels' height above ground (m)
		real(8), dimension(:), allocatable	:: u		! U components (m/s)
		real(8), dimension(:), allocatable	:: v		! V components (m/s)
		real(8), dimension(:), allocatable	:: T		! Temperatures (K)
		real(8), dimension(:), allocatable	:: su2		! var(U) values (m2/s2)
		real(8), dimension(:), allocatable	:: sv2		! var(V) values (m2/s2)
		real(8), dimension(:), allocatable	:: sw2		! var(W) values (m2/s2)
		real(8), dimension(:), allocatable	:: dsw2		! d var(W) / dz (m/s2)
		real(8), dimension(:), allocatable	:: eps		! TKE dissipation rate
		real(8), dimension(:), allocatable	:: alfa		! Langevin equation coefficient
		real(8), dimension(:), allocatable	:: beta		! Langevin equation coefficient
		real(8), dimension(:), allocatable	:: gamma	! Langevin equation coefficient
		real(8), dimension(:), allocatable	:: delta	! Langevin equation coefficient
		real(8), dimension(:), allocatable	:: alfa_u	! Langevin equation coefficient
		real(8), dimension(:), allocatable	:: alfa_v	! Langevin equation coefficient
		real(8), dimension(:), allocatable	:: deltau	! Langevin equation coefficient
		real(8), dimension(:), allocatable	:: deltav	! Langevin equation coefficient
		real(8), dimension(:), allocatable	:: deltat	! Langevin equation coefficient
		! Convenience derived values
		real(8), dimension(:), allocatable	:: Au		! exp(alfa_u*dt)
		real(8), dimension(:), allocatable	:: Av		! exp(alfa_v*dt)
		real(8), dimension(:), allocatable	:: A		! exp(alfa*dt)
		real(8), dimension(:), allocatable	:: B		! exp(beta*dt)
	contains
		procedure	:: clean      => metpClean
		procedure	:: alloc      => metpAlloc
		procedure	:: initialize => metpInitialize
		procedure	:: create     => metpCreate
		procedure	:: evaluate   => metpEvaluate
		procedure	:: dump       => metpDump
	end type MetProfiles


	type MetProfValues
		real(8)	:: rEpoch	! Time stamp of current profile set
		real(8)	:: z		! Levels' height above ground (m)
		real(8)	:: u		! U components (m/s)
		real(8)	:: v		! V components (m/s)
		real(8)	:: T		! Temperatures (K)
		real(8)	:: su2		! var(U) values (m2/s2)
		real(8)	:: sv2		! var(V) values (m2/s2)
		real(8)	:: sw2		! var(W) values (m2/s2)
		real(8)	:: dsw2		! d var(W) / dz (m/s2)
		real(8)	:: eps		! TKE dissipation rate
		real(8)	:: alfa		! Langevin equation coefficient
		real(8)	:: beta		! Langevin equation coefficient
		real(8)	:: gamma	! Langevin equation coefficient
		real(8)	:: delta	! Langevin equation coefficient
		real(8)	:: alfa_u	! Langevin equation coefficient
		real(8)	:: alfa_v	! Langevin equation coefficient
		real(8)	:: deltau	! Langevin equation coefficient
		real(8)	:: deltav	! Langevin equation coefficient
		real(8)	:: deltat	! Langevin equation coefficient
		real(8)	:: Au		! exp(alfa_u*dt)
		real(8)	:: Av		! exp(alfa_v*dt)
		real(8)	:: A		! exp(alfa*dt)
		real(8)	:: B		! exp(beta*dt)
	end type MetProfValues

contains

	function cfgRead(this, iLUN, iLUN1, sFileName) result(iRetCode)

		! Routine arguments
		class(Config), intent(out)		:: this
		integer, intent(in)				:: iLUN
		integer, intent(in)				:: iLUN1
		character(len=*), intent(in)	:: sFileName
		integer							:: iRetCode

		! Locals
		integer				:: iErrCode
		type(IniFile)		:: cfg
		character(len=128)	:: sBuffer
		integer				:: iNumData
		integer				:: iData

		! Assume success (will falsify on failure)
		iRetCode = 0
		this % lIsFull = .false.

		! Get configuration file, and prepare to parse it
		iErrCode = cfg % read(10, sFileName)
		if(iErrCode /= 0) then
			iRetCode = 1
			return
		end if

		! Gather configuration data
		! -1- General
		iErrCode = cfg % getInteger("General", "debug_level", this % iDebugLevel, 0)
		if(iErrCode /= 0) then
			iRetCode = 2
			return
		end if
		iErrCode = cfg % getString("General", "diafile", this % sDiaFile, "")
		if(iErrCode /= 0) then
			iRetCode = 3
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'diag_file' in [General]"
			return
		end if
		! -1- Timing
		iErrCode = cfg % getInteger("Timing", "avgtime", this % iAvgTime, 3600)
		if(iErrCode /= 0) then
			iRetCode = 4
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'avgtime' in [Timing]"
			return
		end if
		iErrCode = cfg % getInteger("Timing", "nstep", this % iNumStep, 360)
		if(iErrCode /= 0) then
			iRetCode = 5
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'nstep' in [Timing]"
			return
		end if
		! -1- Meteo
		iErrCode = cfg % getString("Meteo", "inpfile", this % sMetInpFile, "")
		if(iErrCode /= 0) then
			iRetCode = 6
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'inpfile' in [Meteo]"
			return
		end if
		iErrCode = cfg % getString("Meteo", "outfile", this % sMetOutFile, "")
		if(iErrCode /= 0) then
			iRetCode = 7
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'inpfile' in [Meteo]"
			return
		end if
		iErrCode = cfg % getString("Meteo", "diafile", this % sMetDiaFile, "")
		if(iErrCode /= 0) then
			iRetCode = 8
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'inpfile' in [Meteo]"
			return
		end if
		iErrCode = cfg % getReal8("Meteo", "height", this % rHeight, -9999.9d0)
		if(iErrCode /= 0) then
			iRetCode = 9
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'height' in [Meteo]"
			return
		end if
		iErrCode = cfg % getReal8("Meteo", "z0", this % rZ0, 0.02d0)
		if(iErrCode /= 0) then
			iRetCode = 10
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'z0' in [Meteo]"
			return
		end if
		iErrCode = cfg % getReal8("Meteo", "zr", this % rZr, 10.d0)
		if(iErrCode /= 0) then
			iRetCode = 11
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'zr' in [Meteo]"
			return
		end if
		iErrCode = cfg % getReal8("Meteo", "zt", this % rZt, 2.d0)
		if(iErrCode /= 0) then
			iRetCode = 12
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'zt' in [Meteo]"
			return
		end if
		iErrCode = cfg % getReal8("Meteo", "gamma", this % rGamma, -0.0098d0)
		if(iErrCode /= 0) then
			iRetCode = 13
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'gamma' in [Meteo]"
			return
		end if
		iErrCode = cfg % getInteger("Meteo", "hemisphere", this % iHemisphere, 1)
		if(iErrCode /= 0) then
			iRetCode = 14
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'iHemisphere' in [Meteo]"
			return
		end if
		iErrCode = cfg % getInteger("Output", "nz", this % iNz, -9999)
		if(iErrCode /= 0) then
			iRetCode = 15
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'nz' in [Output]"
			return
		end if
		iErrCode = cfg % getReal8("Output", "dz", this % rDz, -9999.9d0)
		if(iErrCode /= 0) then
			iRetCode = 16
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid 'dz' in [Output]"
			return
		end if

		! Validate configuration data
		! -1- Timing
		if(this % iAvgTime <= 0 .or. this % iAvgTime > 3600 .or. mod(3600, this % iAvgTime) /= 0) then
			iRetCode = 17
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'avgtime' in [Timing]"
			return
		end if
		if(this % iNumStep < 1 .or. mod(this % iAvgTime, this % iNumStep) /= 0) then
			iRetCode = 18
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'nstep' in [Timing]"
			return
		end if
		if(this % iDebugLevel > 1) print *, "metpre:: info: [Timing] section check done"
		if(this % iNz <= 1) then
			iRetCode = 19
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'nz' in [Output]"
			return
		end if
		if(this % rDz <= 0.d0) then
			iRetCode = 20
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'dz' in [Output]"
			return
		end if
		this % zmax = this % rDz * (this % iNz - 1)
		if(this % iDebugLevel > 1) print *, "metpre:: info: [Output] section check done"
		! -1- Meteorological data
		if(this % rHeight < 0.d0) then
			iRetCode = 21
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'height' in [Meteo]"
			return
		end if
		if(this % rZ0 < 0.d0) then
			iRetCode = 22
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'z0' in [Meteo]"
			return
		end if
		if(this % rZr <= 0.d0) then
			iRetCode = 23
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'zr' in [Meteo]"
			return
		end if
		if(this % rZt <= 0.d0) then
			iRetCode = 24
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'zt' in [Meteo]"
			return
		end if
		if(this % rGamma >= 0.d0) then
			iRetCode = 25
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'gamma' in [Meteo]"
			return
		end if
		if(this % iHemisphere < 0 .or. this % iHemisphere > 1) then
			iRetCode = 26
			if(this % iDebugLevel > 0) print *, "metpre:: error: Invalid value of 'iHemisphere' in [Meteo]"
			return
		end if
		iErrCode = this % tMeteo % read(iLUN1, this % sMetInpFile, this % iAvgTime, this % iNumStep, this % sMetOutFile)
		if(iErrCode /= 0) then
			iRetCode = 27
			if(this % iDebugLevel > 0) print *, "metpre:: error: Meteo data not read, with return code ", iErrCode
			return
		end if
		if(this % iDebugLevel > 1) print *, "metpre:: info: [Meteo] section check done"

		! Leave
		this % lIsFull = .true.

	end function cfgRead


	function cfgGetMeteoSize(this) result(iMeteoSize)

		! Routine arguments
		class(Config), intent(in)	:: this
		integer						:: iMeteoSize

		! Locals
		! --none--

		! Get the information piece desired
		if(this % lIsFull) then
			iMeteoSize = size(this % tMeteo % rvExtEpoch)
		else
			iMeteoSize = 0
		end if

	end function cfgGetMeteoSize


	function cfgGetTimeSteps(this) result(iMeteoSize)

		! Routine arguments
		class(Config), intent(in)	:: this
		integer						:: iMeteoSize

		! Locals
		! --none--

		! Get the information piece desired
		if(this % lIsFull) then
			iMeteoSize = size(this % tMeteo % rvEpoch)
		else
			iMeteoSize = 0
		end if

	end function cfgGetTimeSteps


	function cfgGetTimeSubSteps(this) result(iMeteoSize)

		! Routine arguments
		class(Config), intent(in)	:: this
		integer						:: iMeteoSize

		! Locals
		! --none--

		! Get the information piece desired
		if(this % lIsFull) then
			iMeteoSize = size(this % tMeteo % rvExtEpoch) / size(this % tMeteo % rvEpoch)
		else
			iMeteoSize = 0
		end if

	end function cfgGetTimeSubSteps


	function metpClean(this) result(iRetCode)

		! Routine arguments
		class(MetProfiles), intent(out)		:: this
		integer								:: iRetCode

		! Locals
		! --none--

		! Assume success (will falsify on failure)
		iRetCode = 0

		! Clean-up workspace
		if(allocated(this % z))      deallocate(this % z)
		if(allocated(this % u))      deallocate(this % u)
		if(allocated(this % v))      deallocate(this % v)
		if(allocated(this % T))      deallocate(this % T)
		if(allocated(this % su2))    deallocate(this % su2)
		if(allocated(this % sv2))    deallocate(this % sv2)
		if(allocated(this % sw2))    deallocate(this % sw2)
		if(allocated(this % dsw2))   deallocate(this % dsw2)
		if(allocated(this % eps))    deallocate(this % eps)
		if(allocated(this % alfa))   deallocate(this % alfa)
		if(allocated(this % beta))   deallocate(this % beta)
		if(allocated(this % gamma))  deallocate(this % gamma)
		if(allocated(this % delta))  deallocate(this % delta)
		if(allocated(this % alfa_u)) deallocate(this % alfa_u)
		if(allocated(this % alfa_v)) deallocate(this % alfa_v)
		if(allocated(this % deltau)) deallocate(this % deltau)
		if(allocated(this % deltav)) deallocate(this % deltav)
		if(allocated(this % deltat)) deallocate(this % deltat)
		if(allocated(this % Au))     deallocate(this % Au)
		if(allocated(this % Av))     deallocate(this % Av)
		if(allocated(this % A))      deallocate(this % A)
		if(allocated(this % B))      deallocate(this % B)

	end function metpClean


	function metpAlloc(this, iNumData) result(iRetCode)

		! Routine arguments
		class(MetProfiles), intent(out)		:: this
		integer, intent(in)					:: iNumData
		integer								:: iRetCode

		! Locals
		integer		:: iErrCode

		! Assume success (will falsify on failure)
		iRetCode = 0

		! Check parameters
		if(iNumData <= 0) then
			iRetCode = 1
			return
		end if

		! Reserve workspace
		allocate( &
			this % z(iNumData), &
			this % u(iNumData), &
			this % v(iNumData), &
			this % T(iNumData), &
			this % su2(iNumData), &
			this % sv2(iNumData), &
			this % sw2(iNumData), &
			this % dsw2(iNumData), &
			this % eps(iNumData), &
			this % alfa(iNumData), &
			this % beta(iNumData), &
			this % gamma(iNumData), &
			this % delta(iNumData), &
			this % alfa_u(iNumData), &
			this % alfa_v(iNumData), &
			this % deltau(iNumData), &
			this % deltav(iNumData), &
			this % deltat(iNumData), &
			this % Au(iNumData), &
			this % AV(iNumData), &
			this % A(iNumData), &
			this % B(iNumData), &
			stat = iErrCode &
		)
		if(iRetCode /= 0) then
			iRetCode = 2
			return
		end if

	end function metpAlloc


	function metpCreate( &
		this, &
		cfg, &
		i &		! Index of current row in 'met'
	) result(iRetCode)

		! Routine arguments
		class(MetProfiles), intent(out)		:: this
		type(Config), intent(in)			:: cfg
		integer, intent(in)					:: i
		integer								:: iRetCode

		! Locals
		integer	:: n		! Max number of met data
		integer	:: m		! Number of levels
		integer	:: j
		integer	:: iErrCode
		real(8)	:: Ta		! Absolute temperature (K)
		real(8)	:: Pres		! Air pressure (hPa)
		real(8)	:: rc		! rho*Cp
		real(8)	:: wT		! mean(w'T')
		real(8)	:: hL		! 1/L
		real(8)	:: Ts		! Scale temperature (°C)
		real(8)	:: ws		! Deardoff velocity (m/s)
		real(8)	:: C0u
		real(8)	:: C0v
		real(8)	:: C0w
		real(8)	:: C0uu
		real(8)	:: C0vv
		real(8)	:: C0ww
		real(8)	:: ssw2_2
		real(8)	:: dt
		type(DateTime)	:: tStamp
		character(len=23)	:: sTimeStamp

		! Constants
		real(8), parameter	:: K    = 0.4d0		! von Karman constant
		real(8), parameter	:: G    = 9.81d0	! Universal gravity constant
		real(8), parameter	:: P0   = 1013.d0	! Pressure assumed at 0m MSL

		! Assume success (will falsify on failure)
		iRetCode = 0

		! Check critical parameters
		n = size(cfg % tMeteo % rvExtEpoch)
		if(i < 1 .or. i > n) then
			iRetCode = 1
			return
		end if

		! Initialize
		m = cfg % iNz
		iErrCode = this % clean()
		if(iErrCode /= 0) then
			iRetCode = 2
			return
		end if
		iErrCode = this % alloc(m)
		if(iErrCode /= 0) then
			iRetCode = 3
			return
		end if
		this % z = [(cfg % rZ0 + (j-1) * cfg % rDz, j = 1, m)]
		Ta = cfg % tMeteo % rvExtTemp(i) + 273.15d0

		! Assign time stamp
		this % rEpoch = cfg % tMeteo % rvExtEpoch(i)

		! Estimate ground pressure at site
		Pres = P0 * exp(-0.0342d0 * cfg % rHeight / Ta)

		! Estimation of RhoCp and wT (harmless, and not passed as 'met' data to avoid clutter)
		rc = 350.125d0 * Pres / Ta
		wT = cfg % tMeteo % rvExtH0(i) / rc

		! Reciprocal of Obukhov length
		hL = -K*G/Ta * wT / cfg % tMeteo % rvExtUstar(i)**3

		! Scale temperature
		Ts = -wT / cfg % tMeteo % rvExtUstar(i)

		! Deardoff velocity
		ws = wStar(real(Ta,kind=4), real(cfg % tMeteo % rvExtH0(i),kind=4), real(cfg % tMeteo % rvExtZi(i),kind=4))

		! Estimate wind and temperature profiles, based on SL similarity
		iErrCode = WindProfile( &
			cfg % iHemisphere, &
			this % z, &
			cfg % rZr, &
			cfg % tMeteo % rvExtVel(i), &
			cfg % tMeteo % rvExtDir(i), &
			cfg % rZ0, &
			cfg % tMeteo % rvExtZi(i), &
			cfg % tMeteo % rvExtUstar(i), &
			hL, &
			this % u, &
			this % v &
		)
		if(iErrCode /= 0) then
			iRetCode = 4
			return
		end if
		iErrCode = TempProfile( &
			this % z, &
			cfg % rZ0, &
			cfg % rZt, &
			Ta, &
			-cfg % rGamma, &
			cfg % tMeteo % rvExtZi(i), &
			Ts, &
			cfg % tMeteo % rvExtUstar(i), &
			hL, &
			this % T &
		)
		if(iErrCode /= 0) then
			iRetCode = 5
			return
		end if

		! Estimate vertical and horizontal sigmas
		iErrCode = VerticalWindVarProfile( &
			this % z, &
			cfg % tMeteo % rvExtUstar(i), &
			ws, &
			cfg % rZ0, &
			cfg % tMeteo % rvExtZi(i), &
			this % sw2, &
			this % dsw2 &
		)
		if(iErrCode /= 0) then
			iRetCode = 6
			return
		end if
		iErrCode = HorizontalWindVarProfile( &
			this % z, &
			cfg % tMeteo % rvExtUstar(i), &
			ws, &
			cfg % tMeteo % rvExtZi(i), &
			this % su2, &
			this % sv2 &
		)
		if(iErrCode /= 0) then
			iRetCode = 7
			return
		end if

		! TKE dissipation
		iErrCode = TKEDissipationProfile( &
			this % z, &
			cfg % tMeteo % rvExtUstar(i), &
			ws, &
			cfg % rZ0, &
			cfg % tMeteo % rvExtZi(i), &
			this % eps &
		)
		if(iErrCode /= 0) then
			iRetCode = 8
			return
		end if

		! Kolmogorov coefficients, used in further calculations
		iErrCode = KolmogorovConstants(ws, C0u, C0v, C0w)
		if(iErrCode /= 0) then
			print *, iErrCode
			iRetCode = 9
			return
		end if

		! Langevin coefficients and optimal time step (a function
		! of vertical Lagrangian decorrelation time)
		do j = 1, m
			if(ws > 0.) then
				! Convective
				C0uu   = C0u * this % eps(j)
				C0vv   = C0v * this % eps(j)
				C0ww   = C0w * this % eps(j)
				ssw2_2 = 2.d0 * this % sw2(j)
				if(this % z(j) <= cfg % tMeteo % rvExtZi(i)) then
					! Inside the PBL

					! Langevin coefficients for W component
					this % alfa(j)   = this % dsw2(j) / ssw2_2
					this % beta(j)   = -C0ww / ssw2_2
					this % gamma(j)  = 0.5d0 * this % dsw2(j)
					this % delta(j)  = sqrt(C0ww)

					! Optimal time step
					this % deltat(j) = 0.1d0 * ssw2_2 / C0ww

					! Langevin coefficients for U, V component
					this % alfa_u(j) = -C0uu / (2.d0 * this % su2(j))
					this % alfa_v(j) = -C0vv / (2.d0 * this % sv2(j))
					this % deltau(j) = sqrt(C0uu)
					this % deltav(j) = sqrt(C0vv)

				else
					! Above the PBL

					! Langevin coefficients for W component
					this % alfa(j)   = 0.d0
					this % beta(j)   = -C0ww / ssw2_2
					this % gamma(j)  = 0.d0
					this % delta(j)  = sqrt(C0ww)

					! Optimal time step
					this % deltat(j) = 100.d0	! Not used, in reality: just an indication

					! Langevin coefficients for U, V component
					this % alfa_u(j) = -C0uu / (2.d0 * this % su2(j))
					this % alfa_v(j) = -C0vv / (2.d0 * this % sv2(j))
					this % deltau(j) = sqrt(C0uu)
					this % deltav(j) = sqrt(C0vv)

				end if

			else
				! Stable

				! Langevin coefficients for W component
				C0ww             = C0w * this % eps(j)
				ssw2_2           = 2.d0 * this % sw2(j)
				this % alfa(j)   = this % dsw2(j) / ssw2_2
				this % beta(j)   = -C0ww / ssw2_2
				this % gamma(j)  = 0.5d0 * this % dsw2(j)
				this % delta(j)  = sqrt(C0ww)

				! Optimal time step
				this % deltat(j) = 0.1d0 * ssw2_2/C0ww

				! Langevin coefficients for U, V component
				C0uu             = C0u * this % eps(j)
				C0vv             = C0v * this % eps(j)
				this % alfa_u(j) = -C0uu / (2.d0 * this % su2(j))
				this % alfa_v(j) = -C0vv / (2.d0 * this % sv2(j))
				this % deltau(j) = sqrt(C0uu)
				this % deltav(j) = sqrt(C0vv)

			end if

		end do

		! Convenience values
		dt        = real(cfg % iAvgTime, kind=8) / real(cfg % iNumStep, kind=8)
		this % Au = exp(this % alfa_u * dt)
		this % Av = exp(this % alfa_v * dt)
		this % A  = exp(this % alfa * dt)
		this % B  = exp(this % beta * dt)

		! Diagnostic printouts (provisional)
		if(cfg % iDebugLevel >= 3) then
			iErrCode = tStamp % fromEpoch(cfg % tMeteo % rvExtEpoch(i))
			sTimeStamp = tStamp % ToISO()
			print *, "Meteo profiles range report for step at ", sTimeStamp
			print *, "U>      ", minval(this % u), maxval(this % u)
			print *, "V>      ", minval(this % v), maxval(this % v)
			print *, "Vel>    ", minval(sqrt(this % u**2 + this % v**2)), maxval(sqrt(this % u**2 + this % v**2))
			print *, "T>      ", minval(this % T), maxval(this % T)
			print *, "sU2>    ", minval(this % su2), maxval(this % su2)
			print *, "sV2>    ", minval(this % sv2), maxval(this % sv2)
			print *, "sW2>    ", minval(this % sw2), maxval(this % sw2)
			print *, "dsW2>   ", minval(this % dsW2), maxval(this % dsW2)
			print *, "eps>    ", minval(this % eps), maxval(this % eps)
			print *, "alpha>  ", minval(this % alfa), maxval(this % alfa)
			print *, "alphau> ", minval(this % alfa_u), maxval(this % alfa_u)
			print *, "alphav> ", minval(this % alfa_v), maxval(this % alfa_v)
			print *, "beta>   ", minval(this % beta), maxval(this % beta)
			print *, "gamma>  ", minval(this % gamma), maxval(this % gamma)
			print *, "delta>  ", minval(this % delta), maxval(this % delta)
			print *, "deltau> ", minval(this % deltau), maxval(this % deltau)
			print *, "deltav> ", minval(this % deltav), maxval(this % deltav)
			print *, "A>      ", minval(this % A), maxval(this % A)
			print *, "Au>     ", minval(this % Au), maxval(this % Au)
			print *, "Av>     ", minval(this % Av), maxval(this % Av)
			print *, "B>      ", minval(this % B), maxval(this % B)
			print *
		end if

	end function metpCreate


	function metpEvaluate( &
		this, &		! Current meteo profiles
		cfg, &		! Configuration parameters
		zp, &		! Reference height at which to evaluate
		met &
	) result(iRetCode)

		! Routine arguments
		class(MetProfiles), intent(in)		:: this
		type(Config), intent(in)			:: cfg
		real(8), intent(in)					:: zp
		type(MetProfValues), intent(out)	:: met
		integer								:: iRetCode

		! Locals
		integer	:: n
		real(8)	:: zpp
		integer	:: izFrom
		integer	:: izTo

		! Assume success (will falsify on failure
		iRetCode = 0

		! Identify the indices bounding the desired height
		n = size(this % z)
		if(zp <= this % z(1)) then
			izFrom = 1
			izTo   = 1
		elseif(zp >= this % z(n)) then
			izFrom = n
			izTo   = n
		else ! Entry condition: z(1) < zp < z(n)
			izFrom = floor((zp - cfg % rZ0) / cfg % rDz) + 1
			izTo   = ceiling((zp - cfg % rZ0) / cfg % rDz) + 1
			if(izFrom < 1 .or. izFrom > n) then
				print *, 'iZfrom = ', izFrom
				print *, '         ', zp, cfg % rZ0, cfg % rDz
			end if
		end if

		! Evaluate linear interpolation coefficients
		zpp = (zp - this % z(izFrom)) / cfg % rDz

		! Compute linear interpolation
		met % u      = this % u(izFrom)      + zpp * (this % u(izTo)      - this % u(izFrom))
		met % v      = this % v(izFrom)      + zpp * (this % v(izTo)      - this % v(izFrom))
		met % su2    = this % su2(izFrom)    + zpp * (this % su2(izTo)    - this % su2(izFrom))
		met % sv2    = this % sv2(izFrom)    + zpp * (this % sv2(izTo)    - this % sv2(izFrom))
		met % sw2    = this % sw2(izFrom)    + zpp * (this % sw2(izTo)    - this % sw2(izFrom))
		met % dsw2   = this % dsw2(izFrom)   + zpp * (this % dsw2(izTo)   - this % dsw2(izFrom))
		met % eps    = this % eps(izFrom)    + zpp * (this % eps(izTo)    - this % eps(izFrom))
		met % alfa   = this % alfa(izFrom)   + zpp * (this % alfa(izTo)   - this % alfa(izFrom))
		met % beta   = this % beta(izFrom)   + zpp * (this % beta(izTo)   - this % beta(izFrom))
		met % gamma  = this % gamma(izFrom)  + zpp * (this % gamma(izTo)  - this % gamma(izFrom))
		met % delta  = this % delta(izFrom)  + zpp * (this % delta(izTo)  - this % delta(izFrom))
		met % alfa_u = this % alfa_u(izFrom) + zpp * (this % alfa_u(izTo) - this % alfa_u(izFrom))
		met % alfa_v = this % alfa_v(izFrom) + zpp * (this % alfa_v(izTo) - this % alfa_v(izFrom))
		met % deltau = this % deltau(izFrom) + zpp * (this % deltau(izTo) - this % deltau(izFrom))
		met % deltav = this % deltav(izFrom) + zpp * (this % deltav(izTo) - this % deltav(izFrom))
		met % deltat = this % deltat(izFrom) + zpp * (this % deltat(izTo) - this % deltat(izFrom))
		met % Au     = this % Au(izFrom)     + zpp * (this % Au(izTo)     - this % Au(izFrom))
		met % Av     = this % Av(izFrom)     + zpp * (this % Av(izTo)     - this % Av(izFrom))
		met % A      = this % A(izFrom)      + zpp * (this % A(izTo)      - this % A(izFrom))
		met % B      = this % B(izFrom)      + zpp * (this % B(izTo)      - this % B(izFrom))

	end function metpEvaluate


	function metpInitialize(this, tConfig, iLUN) result(iRetCode)

				! Routine arguments
				class(MetProfiles), intent(in)	:: this
				type(Config), intent(in)				:: tConfig
				integer, intent(in)							:: iLUN
				integer													:: iRetCode

				! Locals
				integer	:: iErrCode

				! Assume success (will falsify on failure)
				iRetCode = 0

				! Write profiles
				! -1- Try accessing file and write header
				! -2- Access
				open(iLUN, file=tConfig % sMetOutFile, status='unknown', action='write', access='stream', iostat=iErrCode)
				if(iErrCode /= 0) then
					iRetCode = 2
					return
				end if
				! -2- Header
				! -3- General
				write(iLUN) tConfig % iDebugLevel
				write(iLUN) tConfig % sDiaFile
				write(iLUN) tConfig % iFrameInterval
				write(iLUN) tConfig % sFramePath
				write(iLUN) tConfig % iExecMode
				! -3- Timing
				write(iLUN) tConfig % iAvgTime
				write(iLUN) tConfig % iNumStep
				write(iLUN) tConfig % iNumPart
				write(iLUN) tConfig % iMaxAge
				! -3- Emission
				write(iLUN) tConfig % sStatic
				write(iLUN) tConfig % sDynamic
				! -3- Meteo
				write(iLUN) tConfig % sMetInpFile
				write(iLUN) tConfig % sMetOutFile
				write(iLUN) tConfig % sMetDiaFile
				write(iLUN) tConfig % rHeight
				write(iLUN) tConfig % rZ0
				write(iLUN) tConfig % rZr
				write(iLUN) tConfig % rZt
				write(iLUN) tConfig % rGamma
				write(iLUN) tConfig % iHemisphere
				! -3- Output
				write(iLUN) tConfig % sConcFile
				write(iLUN) tConfig % rX0
				write(iLUN) tConfig % rY0
				write(iLUN) tConfig % iNx
				write(iLUN) tConfig % iNy
				write(iLUN) tConfig % iNz
				write(iLUN) tConfig % rDx
				write(iLUN) tConfig % rDy
				write(iLUN) tConfig % rDz
				! -3- Computed

	end function metpInitialize


	function metpDump(this, iLUN) result(iRetCode)

		! Routine arguments
		class(MetProfiles), intent(in)	:: this
		integer, intent(in)							:: iLUN
		integer													:: iRetCode

		! Locals
		integer	:: iErrCode
		integer	:: i

		! Assume success (will falsify on failure)
		iRetCode = 0

		! -1- Write meteo data part
		do i = 1, size(this % z)
			write(iLUN) &
				this % z(i), &
				this % u(i), this % v(i), &
				this % T(i), &
				this % su2(i), this % sv2(i), this % sw2(i), this % dsw2(i), &
				this % eps(i), &
				this % alfa(i), this % beta(i), this % gamma(i), this % delta(i), &
				this % alfa_u(i), this % alfa_v(i), &
				this % deltau(i), this % deltav(i), this % deltat(i), &
				this % Au(i), this % Av(i), this % A(i), this % B(i)
		end do

	end function metpDump

end module Processing
