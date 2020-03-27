module data_file

    implicit none
    
    private
    
    ! Public interface
    public  :: PartType
    
    type PartType
        integer                             :: iLUN
        integer                             :: iNumData
        integer                             :: iNumMeteoData
        integer                             :: iNumPart
        integer                             :: iNumIteration
        integer                             :: iCurrentTime
        real                                :: rU
        real                                :: rV
        real                                :: rStdDevU
        real                                :: rStdDevV
        real                                :: rCovUV
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        integer, dimension(:), allocatable  :: ivTimeStamp
    contains
        procedure   :: Open
        procedure   :: Read
        procedure   :: Close
    end type PartType
    
contains

    function Open(this, iLUN, sFileName) result(iRetCode)
    
        ! Routine arguments
        class(PartType), intent(inout)  :: this
        integer, intent(in)             :: iLUN
        character(len=*), intent(in)    :: sFileName
        integer                         :: iRetCode
        
        ! Locals
        integer     :: iErrCode
        integer     :: iNumData
        integer     :: iNumMeteoData
        
        ! Assume success (will falsify on failure)
        iRetCode        =  0
        this % iLUN     = -1
        this % iNumPart =  0
        
        ! Try accessing file
        open(iLUN, file=sFileName, action='read', status='old', access='stream', iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        read(iLUN, iostat=iErrCode) iNumData
        if(iErrCode /= 0) then
            iRetCode = 2
            close(iLUN)
            return
        end if
        if(iNumData <= 0) then
            iRetCode = 3
            close(iLUN)
            return
        end if
        read(iLUN, iostat=iErrCode) iNumMeteoData
        if(iErrCode /= 0) then
            iRetCode = 4
            close(iLUN)
            return
        end if
        if(iNumMeteoData <= 0) then
            iRetCode = 5
            close(iLUN)
            return
        end if
        
        ! Inform this file is accessible
        this % iLUN          = iLUN
        this % iNumData      = iNumData
        this % iNumMeteoData = iNumMeteoData
        
    end function Open
    

    function Read(this) result(iRetCode)
    
        ! Routine arguments
        class(PartType), intent(inout)  :: this
        integer                         :: iRetCode
        
        ! Locals
        integer     :: iErrCode
        integer     :: i
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Try gathering a value
        read(this % iLUN, iostat = iErrCode) this % iNumPart
        if(iErrCode /= 0) then
            iRetCode = -1
            return
        end if
        
        if(this % iNumPart > 0) then
        
            ! Reserve storage space
            if(allocated(this % rvX))         deallocate(this % rvX)
            if(allocated(this % rvY))         deallocate(this % rvY)
            if(allocated(this % ivTimeStamp)) deallocate(this % ivTimeStamp)
            allocate(this % rvX(this % iNumPart))
            allocate(this % rvY(this % iNumPart))
            allocate(this % ivTimeStamp(this % iNumPart))
            
            ! Get actual data
            read(this % iLUN, iostat = iErrCode) this % rvX, this % rvY, this % ivTimeStamp
            if(iErrCode /= 0) then
                this % iNumPart = 0
                iRetCode = -1
                deallocate(this % rvX, this % rvY, this % ivTimeStamp)
                return
            end if
            
        end if
        
    end function Read
    

    function Close(this) result(iRetCode)
    
        ! Routine arguments
        class(PartType), intent(inout)  :: this
        integer                         :: iRetCode
        
        ! Locals
        ! -none-
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
    end function Close

end module data_file
