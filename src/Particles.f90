! Particles.f90 - Fortran module incorporating particle dynamics

module Particles

    implicit none
    
    private
    
    ! Public interface
    public  :: ParticlePoolType
    
    ! Data types
    type ParticlePoolType
        integer                             :: iLastPart
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        real, dimension(:), allocatable     :: rvU
        real, dimension(:), allocatable     :: rvV
        logical, dimension(:), allocatable  :: lvIsActive
    contains
        procedure   :: clean
        procedure   :: init
        procedure   :: start
        procedure   :: release
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
        if(allocated(this % lvIsActive)) deallocate(this % lvIsActive)
        
    end subroutine clean
    

    function init(this, iNumPart) result(iRetCode)
    
        ! Routine arguments
        class(ParticlePoolType), intent(out)    :: this
        integer, intent(in)                     :: iNumPart
        integer                                 :: iRetCode
        
        ! Locals
        ! -none-
        
        ! Assume success (will falsify on failure
        iRetCode = 0
        
        ! Check input parameters
        if(iNumPart <= 0) then
            iRetCode = 1
            return
        end if
        
        ! Clean, then reserve workspace
        call this % clean()
        allocate(this % rvX(iNumPart))
        allocate(this % rvY(iNumPart))
        allocate(this % rvU(iNumPart))
        allocate(this % rvV(iNumPart))
        allocate(this % lvIsActive(iNumPart))
    
    end function init
    
    
    function start(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlePoolType), intent(inout)  :: this
        integer                                 :: iRetCode
        
        ! Locals
        ! -none-
        
        ! Assume success (will falsify on failure
        iRetCode = 0
        
        ! Cleanout all particle space
        this % lvIsActive = .false.
        this % rvX        = 0.
        this % rvY        = 0.
        this % rvU        = 0.
        this % rvV        = 0.
        this % iLastPart  = 0
        
    end function start
    
    
    function release(this, iNumToRelease, rU, rV) result(iRetCode)
    
        ! Routine arguments
        class(ParticlePoolType), intent(inout)  :: this
        integer, intent(in)                     :: iNumToRelease
        real, intent(in)                        :: rU
        real, intent(in)                        :: rV
        integer                                 :: iRetCode
        
        ! Locals
        integer :: iStart
        integer :: iStop
        
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
    
        ! Check input parameters
        if(iNumToRelease <= 0) then
            iRetCode = 1
            return
        end if
        if(mod(size(this % lvIsActive), iNumToRelease) /= 0) then
            iRetCode = 2
            return
        end if
        
        ! Emit new particles
        ! -1- Index bounds
        if(this % iLastPart < size(this % lvIsActive)) then
            iStart = this % iLastPart + 1
            iStop  = this % iLastPart + iNumToRelease
        else
            iStart = 1
            iStop  = iNumToRelease
        end if
        this % rvX(iStart:iStop)        = 0.
        this % rvY(iStart:iStop)        = 0.
        this % rvU(iStart:iStop)        = rU
        this % rvV(iStart:iStop)        = rV
        this % lvIsActive(iStart:iStop) = .true.
        
        ! Update last particle index
        this % iLastPart = iStop
        
    end function release

end module Particles
