module data_file

    implicit none
    
    private
    
    ! Public interface
    public  :: PartType
    
    type PartType
        integer                             :: iLUN
        integer                             :: iNumPart
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
        
        ! Assume success (will falsify on failure)
        iRetCode        =  0
        this % iLUN     = -1
        this % iNumPart =  0
        
        ! Clean storage space
        if(allocated(this % rvX))         deallocate(this % rvX)
        if(allocated(this % rvY))         deallocate(this % rvY)
        if(allocated(this % ivTimeStamp)) deallocate(this % ivTimeStamp)
        
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
        
        ! Reserve storage space
        allocate(this % rvX(iNumData))
        allocate(this % rvY(iNumData))
        allocate(this % ivTimeStamp(iNumData))
        
        ! Inform this file is accessible
        this % iLUN = iLUN
        
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
        do i = 1, this % iNumPart
            read(this % iLUN, iostat = iErrCode) &
                this % rvX(i), &
                this % rvY(i), &
                this % ivTimeStamp(i)
            if(iErrCode /= 0) then
                this % iNumPart = 0
                iRetCode = -1
                return
            end if
        end do
        
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
