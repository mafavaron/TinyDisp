module data_file

    implicit none
    
    private
    
    ! Public interface
    public  :: PartType
    
    type PartType
        integer                             :: iLUN
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
        
    end function Open
    

    function Read(this) result(iRetCode)
    
        ! Routine arguments
        class(PartType), intent(inout)  :: this
        integer                         :: iRetCode
        
        ! Locals
        
    end function Read
    

    function Close(this) result(iRetCode)
    
        ! Routine arguments
        class(PartType), intent(inout)  :: this
        integer                         :: iRetCode
        
        ! Locals
        
    end function Close

end module data_file
