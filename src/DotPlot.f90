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

end module DotPlot
