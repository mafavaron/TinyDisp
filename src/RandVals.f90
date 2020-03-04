module RandVals

    implicit none
    
    private
    
    ! Public interface
    public  :: Norm         ! N(0,1)
    public  :: BivarNorm    ! Bivariate normal

contains

    ! Normal generator, after the "Ratio method for normal deviates"
    ! in D.E. Knuth, "The Art of Computer Programming - Vol.2: Seminumerical Algorithms"
    function Norm() result(rNorm)
    
        ! Routine arguments
        real    :: rNorm
        
        ! Locals
        real    :: rU, rV, rX
        
        ! Constants
        real, parameter :: e   = exp(1.)
        real, parameter :: s8e = sqrt(8/e)
        real, parameter :: e4  = 4. * e**0.25
        real, parameter :: e5  = 4. * e**-1.35
        
        do
        
            ! Step R1
            do
                call random_number(rU)
                if(rU > 0.) exit
            end do
            call random_number(rV)
            rX = s8e * (rV - 0.5) / rU
            
            ! Step R2
            if(rX**2 <= 5. - e4 * rU) then
                rNorm = rX
                return
            end if
            
            ! Step R3
            if(rX**2 >= e5/rU + 1.4) cycle
            
            ! Step R4
            if(rX**2 <= -4./log(rU)) then
                rNorm = rX
                return
            end if
            
        end do
        
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

