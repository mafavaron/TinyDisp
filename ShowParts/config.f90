module Config

    implicit none
    
    private
    
    ! Public interface
    public  :: ConfigType
    
    ! Data types
    
    type ConfigType
        real                :: rEdgeLength
        character(len=256)  :: sParticlesFile
    contains
        procedure           :: Read
    end type ConfigType
    
contains

    function Read(this, iLUN, sConfigFile) result(iRetCode)
    
        ! Routine arguments
        class(ConfigType), intent(out)  :: this
        integer, intent(in)             :: iLUN
        character(len=256)              :: sConfigFile
        integer                         :: iRetCode
        
        ! Locals
        integer             :: iErrCode
        real                :: rEdgeLength
        character(len=256)  :: sParticlesFile
        namelist/cfg/ rEdgeLength, sParticlesFile
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Get configuration from file, assuming NAMELIST form
        open(iLUN, file=sConfigFile, status="old", action="read", iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        read(iLUN, nml=cfg, iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 2
            close(iLUN)
            return
        end if
        close(iLUN)
        
        ! Save configuration data to data type members
        this % rEdgeLength    = rEdgeLength
        this % sParticlesFile = sParticlesFile
        
    end function Read

end module Config
