! Module for accessing TinyDisp own configuration
!
! By: Patrizia M. ("Patti") Favaron
!
module Config

    use Calendar

    implicit none
    
    type ConfigType
        ! -1- General
        real                :: rTimeStep        ! In seconds, strictly positive
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
        real                :: rTimeStep
        real                :: rEdgeLength
        character(len=256)  :: sBuffer
        character(len=256)  :: sMeteoFile
        logical             :: lIsFile
        integer             :: iNumLines
        integer             :: iLine
        namelist /configuration/ &
            rTimeStep, &
            rEdgeLength, &
            sMeteoFile
        ! -1- From meteo file
        integer             :: iCurTime
        real                :: rU
        real                :: rV
        real                :: rStdDevU
        real                :: rStdDevV
        real                :: rCovUV
        integer             :: iYear, iMonth, iDay, iHour, iMinute, iSecond
        
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
        if(rTimeStep <= 0.) then
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
            if(iErrCode > 0) exit
            read(sBuffer(1:19), "(i4,5(1x,i2))", iostat=iErrCode) iYear, iMonth, iDay, iHour, iMinute, iSecond
            if(iErrCode /= 0) exit
            read(sBuffer(20:), *, iostat=iErrCode) rU, rV, rStdDevU, rStdDevV, rCovUV
            if(iErrCode /= 0) exit
            iNumLines = iNumLines + 1
        end do
        if(iNumLines <= 0) then
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
            read(sBuffer(20:), *) &
                this % rvU(iLine), &
                this % rvV(iLine), &
                this % rvStdDevU(iLine), &
                this % rvStdDevV(iLine), &
                this % rvCovUV(iLine)
        end do
        close(iLUN)
        
        ! Form all what remains of configuration, and declare it valid
        this % rTimeStep   = rTimeStep
        this % rEdgeLength = rEdgeLength
        this % sMeteoFile  = sMeteoFile
        this % lIsValid    = .true.
        
    end function gather

end module Config
