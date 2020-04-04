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
        logical                             :: lTwoDimensional
        integer                             :: iNextPart
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        real, dimension(:), allocatable     :: rvZ
        real, dimension(:), allocatable     :: rvU
        real, dimension(:), allocatable     :: rvV
        real, dimension(:), allocatable     :: rvW
        integer, dimension(:), allocatable  :: ivTimeStampAtBirth
    contains
        procedure   :: Create
        procedure   :: Emit
    end type ParticlesPoolType
    
contains

    function Create(this, iNumParts, lTwoDimensional) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesPoolType), intent(inout) :: this
        integer, intent(in)                     :: iNumParts
        logical, intent(in)                     :: lTwoDimensional
        integer                                 :: iRetCode
        
        ! Locals
        ! --none--
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Check parameters
        if(iNumParts <= 0) then
            iRetCode = 1
            return
        end if
        
        ! Reserve workspace
        if(allocated(this % ivTimeStampAtBirth)) allocate(this % ivTimeStampAtBirth(iNumParts))
        if(allocated(this % rvX))                allocate(this % rvX(iNumParts))
        if(allocated(this % rvY))                allocate(this % rvY(iNumParts))
        if(allocated(this % rvZ))                allocate(this % rvZ(iNumParts))
        if(allocated(this % rvU))                allocate(this % rvU(iNumParts))
        if(allocated(this % rvV))                allocate(this % rvV(iNumParts))
        if(allocated(this % rvW))                allocate(this % rvW(iNumParts))
        
        ! Initialize with relevant values
        this % ivTimeStampAtBirth = -1  ! Meaning "inactive"
        this % iNextPart          =  1
    
    end function Create


    function Emit(this, iNumNewParts, iTimeStamp, rU, rV, rW) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesPoolType), intent(inout) :: this
        integer, intent(in)                     :: iNumNewParts
        integer, intent(in)                     :: iTimeStamp
        real, intent(in)                        :: rU
        real, intent(in)                        :: rV
        real, intent(in)                        :: rW
        integer                                 :: iRetCode
        
        ! Locals
        ! --none--
        
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
        if(mod(size(this % ivTimeStampAtBirth), iNumNewParts) /= 0) then
            iRetCode = 3
            return
        end if
        
        ! Emit new particles
        this % ivTimeStampAtBirth(this % iNextPart : this % iNextPart + iNumNewParts) = iTimeStamp
        this % rvX(this % iNextPart : this % iNextPart + iNumNewParts)                = 0.
        this % rvY(this % iNextPart : this % iNextPart + iNumNewParts)                = 0.
        this % rvZ(this % iNextPart : this % iNextPart + iNumNewParts)                = 0.
        this % rvU(this % iNextPart : this % iNextPart + iNumNewParts)                = rU
        this % rvV(this % iNextPart : this % iNextPart + iNumNewParts)                = rV
        this % rvW(this % iNextPart : this % iNextPart + iNumNewParts)                = rW
        
        ! Update next particle index
        this % iNextPart = this % iNextPart + iNumNewParts
        if(this % iNextPart > size(this % ivTimeStampAtBirth)) this % iNextPart = 1
    
    end function Emit

end module Particles
