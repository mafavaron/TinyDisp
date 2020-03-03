module DotPlot

    use appgraphics
    
    implicit none
    
    private
    
    ! Public interface
    public  :: PointType
    
    ! Data types
    type PointType
        real    :: xMultiplier
        real    :: xOffset
        real    :: yMultiplier
        real    :: yOffset
    contains
        procedure   :: init
        procedure   :: show
    end type PointType
    
contains

    function init(this, iWidth, iHeight, xMin, xMax, yMin, yMax) result(iRetCode)
    
        ! Routine arguments
        class(PointType), intent(out)   :: this
        integer, intent(in)             :: iWidth
        integer, intent(in)             :: iHeight
        real, intent(in)                :: xMin
        real, intent(in)                :: xMax
        real, intent(in)                :: yMin
        real, intent(in)                :: yMax
        integer                         :: iRetCode
        
        ! Locals
        integer :: iMin
        integer :: iMax
        integer :: jMin
        integer :: jMax
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Check parameters
        if(iWidth <= 0 .or. iHeight <= 0) then
            iRetCode = 1
            return
        end if
        if(xMin >= xMax .or. yMin >= yMax) then
            iRetCode = 2
            return
        end if
        
        ! Set the extremal viewport coordinates
        iMin = 0
        iMax = iWidth - 1
        jMin = iHeight - 1
        jMax = 0
        
        ! Compute the conversion coefficients
        this % xMultiplier = (iMax - iMin)/(xMax - xMin)
        this % xOffset     = iMin - this % xMultiplier
        this % yMultiplier = (jMax - jMin)/(yMax - yMin)
        this % yOffset     = jMin - this % yMultiplier
        
    end function init
    
    
    function show(this, rvX, rvY) result(iRetCode)
    
        ! Routine arguments
        class(PointType), intent(out)   :: this
        real, dimension(:), intent(in)  :: rvX
        real, dimension(:), intent(in)  :: rvY
        integer                         :: iRetCode
        
        ! Locals
        integer, dimension(:), allocatable  :: ivX
        integer, dimension(:), allocatable  :: ivY
        integer                             :: i
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Reserve workspace
        allocate(ivX(size(rvX)), ivY(size(rvY)))
        
        ! Convert floating point coordinates to integer form
        ivX = nint(rvX * this % xMultiplier + this % xOffset)
        ivY = nint(rvY * this % yMultiplier + this % yOffset)
        
        ! Plot points
        do i = 1, size(rvX)
            call putpixel(ivX(i), ivY(i), LIGHTBLUE)
        end do
        
        ! Leave
        deallocate(ivX, ivY)
        call swapbuffers()
        
    end function show

end module DotPlot
