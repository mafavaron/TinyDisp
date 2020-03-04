module RandVals

    implicit none
    
    private
    
    ! Public interface
    public  :: Norm         ! N(0,1)
    public  :: BivarNorm    ! Bivariate normal

contains

    ! Normal generator
    function Norm() result(rNorm)
    end function Norm
    

    ! Bivariate normal generator
    subroutine NormVals(rvV1, rvV2, rMu1, rMu2, rSigma1, rSigma2, rRho)
    
        ! Routine arguments
        real, dimension(:), intent(out) :: rvV1
        real, dimension(:), intent(out) :: rvV2
        real, intent(in)                :: rMu1
        real, intent(in)                :: rMu2
        real, intent(in)                :: rSigma1
        real, intent(in)                :: rSigma2
        real, intent(in)                :: rRho
        
        ! Locals
        real    :: rLambda
        real    :: rNu
        real    :: rX1
        integer :: i
        
        ! Compute auxiliary values
        rLambda = (rSigma2 / rSigma1) * rRho
        rNu     = sqrt( (1. - rRho*2) * rSigma2**2)
        
        ! Calculate random values
        do i = 1, size(rvV1)
            rX1 = rMu1 + rSigma1 * Norm()
            rvV1(i) = rX1
            rvV2(i) = rMu2 + rLambda * (rX1 - rMu1) + rNu * Norm()
        end do
        
    end subroutine BivarNorm

end module RandVals
