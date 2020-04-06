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
        real, dimension(:), allocatable     :: rvN1
        real, dimension(:), allocatable     :: rvN2
        real, dimension(:), allocatable     :: rvN3
        integer, dimension(:), allocatable  :: ivTimeStampAtBirth
    contains
        procedure   :: Create
        procedure   :: Emit
        procedure   :: Move
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
        if(allocated(this % ivTimeStampAtBirth)) deallocate(this % ivTimeStampAtBirth)
        if(allocated(this % rvX))                deallocate(this % rvX)
        if(allocated(this % rvY))                deallocate(this % rvY)
        if(allocated(this % rvZ))                deallocate(this % rvZ)
        if(allocated(this % rvU))                deallocate(this % rvU)
        if(allocated(this % rvV))                deallocate(this % rvV)
        if(allocated(this % rvW))                deallocate(this % rvW)
        if(allocated(this % rvN1))               deallocate(this % rvN1)
        if(allocated(this % rvN2))               deallocate(this % rvN2)
        if(allocated(this % rvN3))               deallocate(this % rvN3)
        allocate(this % ivTimeStampAtBirth(iNumParts))
        allocate(this % rvX(iNumParts))
        allocate(this % rvY(iNumParts))
        allocate(this % rvZ(iNumParts))
        allocate(this % rvU(iNumParts))
        allocate(this % rvV(iNumParts))
        allocate(this % rvW(iNumParts))
        allocate(this % rvN1(iNumParts))
        allocate(this % rvN2(iNumParts))
        allocate(this % rvN3(iNumParts))
        
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
        this % ivTimeStampAtBirth(this % iNextPart : this % iNextPart + iNumNewParts - 1) = iTimeStamp
        this % rvX(this % iNextPart : this % iNextPart + iNumNewParts - 1)                = 0.
        this % rvY(this % iNextPart : this % iNextPart + iNumNewParts - 1)                = 0.
        this % rvZ(this % iNextPart : this % iNextPart + iNumNewParts - 1)                = 0.
        this % rvU(this % iNextPart : this % iNextPart + iNumNewParts - 1)                = rU
        this % rvV(this % iNextPart : this % iNextPart + iNumNewParts - 1)                = rV
        this % rvW(this % iNextPart : this % iNextPart + iNumNewParts - 1)                = rW
        
        ! Update next particle index
        this % iNextPart = this % iNextPart + iNumNewParts
        if(this % iNextPart > size(this % ivTimeStampAtBirth)) this % iNextPart = 1
    
    end function Emit
    
    
    function Move(this, rU, rV, rW, rUU, rVV, rWW, rUV, rUW, rVW, rDeltaT, rInertia) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesPoolType), intent(inout) :: this
        real, intent(in)                        :: rU
        real, intent(in)                        :: rV
        real, intent(in)                        :: rW
        real, intent(in)                        :: rUU
        real, intent(in)                        :: rVV
        real, intent(in)                        :: rWW
        real, intent(in)                        :: rUV
        real, intent(in)                        :: rUW
        real, intent(in)                        :: rVW
        real, intent(in)                        :: rDeltaT
        real, intent(in)                        :: rInertia
        integer                                 :: iRetCode
        
        ! Locals
        integer :: iErrCode
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Generate tri-variate normal deviates ("mean wind + turbulence")
        iErrCode = MultiNorm(rU, rV, rW, rUU, rVV, rWW, rUV, rUW, rVW, this % rvN1, this % rvN2, this % rvN3)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        
        ! Update wind speed components
        this % rvU = rInertia * this % rvU + (1.-rInertia) * this % rvN1
        this % rvV = rInertia * this % rvV + (1.-rInertia) * this % rvN2
        this % rvW = rInertia * this % rvW + (1.-rInertia) * this % rvN3
        
        ! Update position
        this % rvX = this % rvX + rDeltaT * this % rvU
        this % rvY = this % rvY + rDeltaT * this % rvV
        this % rvZ = this % rvZ + rDeltaT * this % rvW
        
        ! Manage reflection at ground
        where(this % rvZ < 0.)
            this % rvZ = -this % rvZ
            this % rvW = -this % rvW
        end where
    
    end function Move

end module Particles
