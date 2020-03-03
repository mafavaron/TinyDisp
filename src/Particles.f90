! Particles.f90 - Fortran module incorporating particle dynamics

module Particles

    implicit none
    
    private
    
    ! Public interface
    public  :: ParticlePoolType
    
    ! Data types
    type ParticlePoolType
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        real, dimension(:), allocatable     :: rvU
        real, dimension(:), allocatable     :: rvV
        real, dimension(:), allocatable     :: rvDelta
        logical, dimension(:), allocatable  :: lvIsActive
    contains
        procedure   :: clean
        procedure   :: init
    end type ParticlePoolType
    
contains

    subroutine clean(this)
    
        ! Routine arguments
        class(ParticlePoolType), intent(out)    :: this
        
        ! Locals
        ! -none-
        
        ! Reclaim workspace, if any
        if(allocated(this % rvX))        deallocate(this % rvX)
        if(allocated(this % rvY))        deallocate(this % rvY)
        if(allocated(this % rvU))        deallocate(this % rvU)
        if(allocated(this % rvV))        deallocate(this % rvV)
        if(allocated(this % rvDelta))    deallocate(this % rvDelta)
        if(allocated(this % lvIsActive)) deallocate(this % lvIsActive)
        
    end subroutine clean
    

    function init(this, iNumPart) result(iRetCode)
    
        ! Routine arguments
        class(ParticlePoolType), intent(out)    :: this
        integer, intent(in)                     :: iNumPart
        integer                                 :: iRetCode
        
        ! Locals
        
        ! Assume success (will falsify on failure
        iRetCode = 0
        
        ! Clean, then reserve workspace
        call this % clean()
        allocate(this % rvX(iNumPart))
        allocate(this % rvY(iNumPart))
        allocate(this % rvU(iNumPart))
        allocate(this % rvV(iNumPart))
        allocate(this % rvDelta(iNumPart))
        allocate(this % lvIsActive(iNumPart))
    
    end function init

end module Particles
