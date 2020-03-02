! Module for accessing TinyDisp own configuration
!
! By: Patrizia M. ("Patti") Favaron
!
module Config

    use Calendar

    implicit none
    
    type ConfigType
        ! -1- General
        integer             :: iTimeStep        ! In seconds, strictly positive
        real                :: rEdgeLength      ! In metres
        character(len=256)  :: sMeteoFile       ! The desired one
        logical             :: lIsValid         ! Use data type only if .TRUE.
        ! -1- Meteorology
        integer, dimension(:), allocatable  :: ivTimeStamp
        real, dimension(:), allocatable     :: rvU
        real, dimension(:), allocatable     :: rvV
        real, dimension(:), allocatable     :: rvStdDevU
        real, dimension(:), allocatable     :: rvStdDevV
        real, dimension(:), allocatable     :: rvCovUV
    contains
        procedure           :: gather           ! Gets configuration from a NAMELIST file
        procedure           :: get_meteo        ! Get meteo data
    end type ConfigType
    
contains

    function gather(this, iLUN, sCfgFile) result(iRetCode)
    
        ! Routine arguments
        class(ConfigType), intent(out)  :: this
        integer, intent(in)             :: iLUN
        character(len=*), intent(in)    :: sCfgFile
        integer                         :: iRetCode
    
        ! Locals
        ! -1- Just variables
        integer             :: iErrCode
        integer             :: iTimeStep
        real                :: rEdgeLength
        character(len=256)  :: sBuffer
        character(len=256)  :: sMeteoFile
        logical             :: lIsFile
        integer             :: iNumLines
        integer             :: iLine
        namelist /configuration/ iTimeStep, rEdgeLength, sMeteoFile
        ! -1- From meteo file
        integer             :: i
        integer             :: iCurTime
        real                :: rU
        real                :: rV
        real                :: rStdDevU
        real                :: rStdDevV
        real                :: rCovUV
        integer             :: iYear, iMonth, iDay, iHour, iMinute, iSecond
        integer             :: iTimeDelta
        integer             :: iNewTimeDelta
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Get data from Namelist file
        open(iLUN, file = sCfgFile, action='read', status='old', iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        read(iLUN, nml=configuration, iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 2
            close(iLUN)
            return
        end if
        close(iLUN)
        
        ! Validate general configuration
        this % lIsValid = .false.
        if(iTimeStep <= 0) then
            print *, "cfg> Invalid time step: value is zero or negative, should be positive"
            iRetCode = 3
        end if
        if(rEdgeLength <= 0.) then
            print *, "cfg> Invalid edge length: value is zero or negative, should be positive"
            iRetCode = 4
        end if
        inquire(file = sMeteoFile, exist = lIsFile)
        if(.not.lIsFile) then
            print *, "cfg> Invalid met file: file does not exist"
            iRetCode = 5
        end if
        if(iRetCode > 0) return
        
        ! Get meteo data
        open(iLUN, file=sMeteoFile, status='old', action='read', iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 6
            return
        end if
        read(iLUN, "(a)", iostat=iErrCode) sBuffer
        if(iErrCode /= 0) then
            iRetCode = 7
            close(iLUN)
            return
        end if
        iNumLines = 0
        do
            read(iLUN, "(a)", iostat=iErrCode) sBuffer
            if(iErrCode /= 0) exit
            read(sBuffer(1:19), "(i4,5(1x,i2))", iostat=iErrCode) iYear, iMonth, iDay, iHour, iMinute, iSecond
            if(iErrCode /= 0) exit
            do i = 20, len_trim(sBuffer)
                if(sBuffer(i:i) == ',') sBuffer(i:i) = ' '
            end do
            read(sBuffer(20:), *, iostat=iErrCode) rU, rV, rStdDevU, rStdDevV, rCovUV
            if(iErrCode /= 0) exit
            iNumLines = iNumLines + 1
        end do
        if(iNumLines <= 1) then
            iRetCode = 8
            close(iLUN)
            return
        end if
        rewind(iLUN)
        if(allocated(this % ivTimeStamp)) deallocate(this % ivTimeStamp)
        allocate(this % ivTimeStamp(iNumLines))
        if(allocated(this % rvU))         deallocate(this % rvU)
        allocate(this % rvU(iNumLines))
        if(allocated(this % rvV))         deallocate(this % rvV)
        allocate(this % rvV(iNumLines))
        if(allocated(this % rvStdDevU))   deallocate(this % rvStdDevU)
        allocate(this % rvStdDevU(iNumLines))
        if(allocated(this % rvStdDevV))   deallocate(this % rvStdDevV)
        allocate(this % rvStdDevV(iNumLines))
        if(allocated(this % rvCovUV))     deallocate(this % rvCovUV)
        allocate(this % rvCovUV(iNumLines))
        read(iLUN, "(a)") sBuffer
        do iLine = 1, iNumLines
            read(iLUN, "(a)") sBuffer
            read(sBuffer(1:19), "(i4,5(1x,i2))") iYear, iMonth, iDay, iHour, iMinute, iSecond
            call PackTime(this % ivTimeStamp(iLine), iYear, iMonth, iDay, iHour, iMinute, iSecond)
            do i = 20, len_trim(sBuffer)
                if(sBuffer(i:i) == ',') sBuffer(i:i) = ' '
            end do
            read(sBuffer(20:), *) &
                this % rvU(iLine), &
                this % rvV(iLine), &
                this % rvStdDevU(iLine), &
                this % rvStdDevV(iLine), &
                this % rvCovUV(iLine)
        end do
        close(iLUN)
        
        ! Check the meteorological data is valid, that is, with at least
        ! two data records (this has been already tested - iRetCode = 6), and with equally-spaced time stamps
        ! monotonically increasing
        iTimeDelta = this % ivTimeStamp(2) - this % ivTimeStamp(1)
        if(iTimeDelta <= 0) then
            iRetCode = 9
            return
        end if
        do iLine = 3, iNumLines
            iNewTimeDelta = this % ivTimeStamp(iLine) - this % ivTimeStamp(iLine - 1)
            if(iNewTimeDelta /= iTimeDelta) then
                iRetCode = 10
                return
            end if
        end do
        ! Post-condition: time stamps are monotonically increasing, and equally spaced in time
        
        ! Form all what remains of configuration, and declare it valid
        this % iTimeStep   = iTimeStep
        this % rEdgeLength = rEdgeLength
        this % sMeteoFile  = sMeteoFile
        this % lIsValid    = .true.
        
    end function gather
    
    
    function get_meteo(this, ivTimeStamp, rvU, rvV, rvStdDevU, rvStdDevV, rvCovUV) result(iRetCode)
    
        ! Routine arguments
        class(ConfigType), intent(in)                   :: this
        integer, dimension(:), allocatable, intent(out) :: ivTimeStamp
        real, dimension(:), allocatable, intent(out)    :: rvU
        real, dimension(:), allocatable, intent(out)    :: rvV
        real, dimension(:), allocatable, intent(out)    :: rvStdDevU
        real, dimension(:), allocatable, intent(out)    :: rvStdDevV
        real, dimension(:), allocatable, intent(out)    :: rvCovUV
        integer                                         :: iRetCode
        
        ! Locals
        integer, dimension(:), allocatable  :: ivTimeIndex
        integer, dimension(:), allocatable  :: ivTimeShift
        integer :: iMinTimeStamp
        integer :: iMaxTimeStamp
        integer :: iNumTimes
        integer :: iDeltaTime
        integer :: i
        integer :: j
        real    :: rFactor
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Check something can be done
        if(.not. this % lIsValid) then
            iRetCode = 1
            return
        end if
        
        ! Compute the time span, and use it to derive the length of the meteo files
        iMinTimeStamp = minval(this % ivTimeStamp)
        iMaxTimeStamp = maxval(this % ivTimeStamp)
        iNumTimes = (iMaxTimeStamp - iMinTimeStamp) / this % iTimeStep + 1
        
        ! Compute the time step in input meteo data
        iDeltaTime = this % ivTimeStamp(2) - this % ivTimeStamp(1)
        
        ! Reserve space for output values
        if(allocated(ivTimeStamp)) deallocate(ivTimeStamp)
        if(allocated(rvU))         deallocate(rvU)
        if(allocated(rvV))         deallocate(rvV)
        if(allocated(rvStdDevU))   deallocate(rvStdDevU)
        if(allocated(rvStdDevV))   deallocate(rvStdDevV)
        if(allocated(rvCovUV))     deallocate(rvCovUV)
        allocate(ivTimeStamp(iNumTimes))
        allocate(ivTimeIndex(iNumTimes))
        allocate(ivTimeShift(iNumTimes))
        allocate(rvU(iNumTimes))
        allocate(rvV(iNumTimes))
        allocate(rvStdDevU(iNumTimes))
        allocate(rvStdDevV(iNumTimes))
        allocate(rvCovUV(iNumTimes))
        
        ! Generate output time stamps
        ivTimeStamp = [(iMinTimeStamp + this % iTimeStep * (i - 1), i = 1, iNumTimes)]
        
        ! Convert the time stamps to time indices, and to displacements to be used in
        ! linear interpolation sampling of meteorological data
        ivTimeIndex = (ivTimeStamp - ivTimeStamp(1)) / iDeltaTime + 1
        ivTimeShift = ivTimeStamp - this % ivTimeStamp(ivTimeIndex)
        
        ! Interpolate meteorological values
        do i = 1, iNumTimes
            j = ivTimeIndex(i)
            if(ivTimeShift(i) > 0) then
                rFactor      = ivTimeShift(i) / float(iDeltaTime)
                rvU(i)       = this % rvU(j)       + (this % rvU(j+1)       - this % rvU(j)) * rFactor
                rvV(i)       = this % rvV(j)       + (this % rvV(j+1)       - this % rvV(j)) * rFactor
                rvStdDevU(i) = this % rvStdDevU(j) + (this % rvStdDevU(j+i) - this % rvStdDevU(j)) * rFactor
                rvStdDevV(i) = this % rvStdDevV(j) + (this % rvStdDevV(j+1) - this % rvStdDevV(j)) * rFactor
                rvCovUV(i)   = this % rvCovUV(j)   + (this % rvCovUV(j+1)   - this % rvCovUV(j)) * rFactor
            else
                rvU(i)       = this % rvU(j)
                rvV(i)       = this % rvV(j)
                rvStdDevU(i) = this % rvStdDevU(j)
                rvStdDevV(i) = this % rvStdDevV(j)
                rvCovUV(i)   = this % rvCovUV(j)
            end if
        end do
        
        ! Leave
        deallocate(rvTimeShift)
        deallocate(ivTimeIndex)
        
    end function get_meteo

end module Config
