! Module TinyDispFiles
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module TinyDispFiles

    implicit none
    
    private
    
    ! Public interface
    public  :: ParticlesFileType
    
    ! Data types
    
    type ParticlesFileType
        logical                             :: lTwoDimensional
        integer                             :: iLUN
        integer, dimension(:), allocatable  :: ivTimeStamp
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        real, dimension(:), allocatable     :: rvZ
    contains
        procedure open  => prtOpen
        procedure read  => prtRead
        procedure close => prtClose
    end type ParticlesFileType
    
contains

    function prtOpen(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(out)   :: this
        integer                                 :: iRetCode
        
        ! Locals
        integer :: iLUN
        integer :: iErrCode
        integer :: iMaxPart
        integer :: iNumPart
        
        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Initialize structure
        this % lTwoDimensional = .false.
        
        ! Access file
        open(newunit=iLUN, status='old', action='read', access='stream', iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        
        ! Get heading information
        read(iLUN, iostat=iErrCode) iMaxPart, iNumPart
        if(iErrCode /= 0) then
            iRetCode = 2
            close(iLUN)
            return
        end if
        
        ! Store intermediates to structure, for any later use
        this % iLUN = iLUN
        if(iMaxPart < 0) then
            iMaxPart = -iMaxPart
        else
            this % lTwoDimensional = .true.
        end if
        
        ! Reserve storage space, if needed
        if(iNumPart <= 0) then
            iRetCode = 3
            close(iLUN)
            return
        end if
        if(allocated(this % ivTimeStamp)) deallocate(this % ivTimeStamp)
        if(allocated(this % rvX))         deallocate(this % rvX)
        if(allocated(this % rvY))         deallocate(this % rvY)
        if(allocated(this % rvZ))         deallocate(this % rvZ)
        allocate(this % ivTimeStamp(iNumPart))
        allocate(this % rvX(iNumPart))
        allocate(this % rvY(iNumPart))
        allocate(this % rvZ(iNumPart))
        
    end function prtOpen


    function prtRead(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(inout) :: this
        integer                                 :: iRetCode
        
        ! Locals
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
    end function prtRead


    function prtClose(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(inout) :: this
        integer                                 :: iRetCode
        
        ! Locals
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
    end function prtClose

end module TinyDispFiles
