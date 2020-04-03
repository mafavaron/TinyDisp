! Particles - Fortran module for particles generation and dynamics.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module Particles

    use NormalDeviates
    
    implicit none
    
    private
    
    ! Public interface
    public  :: ParticlesPoolType
    
    ! Data types
    type ParticlesPoolType
        integer                             :: iNextPart
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        real, dimension(:), allocatable     :: rvZ
        real, dimension(:), allocatable     :: rvU
        real, dimension(:), allocatable     :: rvV
        real, dimension(:), allocatable     :: rvW
        integer, dimension(:), allocatable  :: ivTimeStampAtBirth
    contains
        procedure   :: Emit
    end type ParticlesPoolType
    
contains

    function Emit(this, iNumNewParts, rU, rV) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesPoolType), intent(inout) :: this
        integer, intent(in)                     :: iNumNewParts
        real, intent(in)                        :: rU
        real, intent(in)                        :: rV
        integer                                 :: iRetCode
        
        ! Locals
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Check parameters
        if(.not.allocated(this % ivTimeStampAtBirth)) then
            iRetCode = 1
            return
        end if
        if(iNumNewParts <= 0) then
            iRetCode = 2
            return
        end if
        
        ! Emit new particles
        
    
    end function Emit

end module Particles
