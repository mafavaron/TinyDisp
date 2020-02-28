! Module for accessing TinyDisp own configuration
!
! By: Patrizia M. ("Patti") Favaron
!
module Config

    implicit none
    
    type ConfigType
        real                :: rTimeStep        ! In seconds, strictly positive
        real                :: rEdgeLength      ! In metres
        character(len=256)  :: sMeteoFile       ! The desired one
        logical             :: lIsValid         ! Use data type only if .TRUE.
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
        real                :: rTimeStep
        real                :: rEdgeLength
        character(len=256)  :: sMeteoFile
        integer             :: iErrCode
        logical             :: lIsFile
        namelist /configuration/ &
            rTimeStep, &
            rEdgeLength, &
            sMeteoFile
        
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
        
        ! Validate configuration
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
        
        ! Form configuration, and declare it valid
        this % rTimeStep   = rTimeStep
        this % rEdgeLength = rEdgeLength
        this % sMeteoFile  = sMeteoFile
        this % lIsValid    = .true.
        
    end function gather

end module Config
