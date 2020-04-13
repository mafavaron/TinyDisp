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
        procedure open
        procedure read
        procedure close
    end type ParticlesFileType
    
contains

    function open(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(out)   :: this
        integer                                 :: iRetCode
        
        ! Locals
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
    end function open


    function read(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(inout) :: this
        integer                                 :: iRetCode
        
        ! Locals
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
    end function read


    function close(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(inout) :: this
        integer                                 :: iRetCode
        
        ! Locals
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
    end function close

end module TinyDispFiles
