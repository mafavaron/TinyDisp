! NormalDeviates - Fortran module for generating univariate and three-variate
!                  random deviates.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module NormalDeviates

	implicit none
	
	private
	
	! Public interface
	public	:: Norm

contains

	! Credit: this routine has been obtained by refactoring of function "random_normal"
	! by Alan Miller (CSIRO Division of Mathematics & Statistics), module 'random.f90'
	function Norm(rvNormValues) result(iRetCode)
	
		! Routine arguments
		real, dimension(:), intent(inout)	:: rvNormValues
		integer								:: iRetCode

		! Locals
		real	:: u, v, x, y, q
		integer	:: i
		
		! Constants
		real, parameter	:: s  =  0.449871
		real, parameter	:: t  = -0.386595
		real, parameter	:: a  =  0.19600
		real, parameter	:: b  =  0.25472
		real, parameter	:: r1 =  0.27597
		real, parameter	:: r2 =  0.27846
		
		! Assume success (will falsify on failure)
		iRetCode = 0

		! Main loop
		if(size(rvNormValues) <= 0) then
			iRetCode = 1
			return
		end if
		do i = 1, size(rvNormValues)

			! Important loop: generate P = (u,v) uniform in rectangle enclosing
			! acceptance region, and iterate until point is really within
			! the acceptance region
			do
			
				! Get uniform deviates using the compiler's generator
				call RANDOM_NUMBER(u)
				call RANDOM_NUMBER(v)
				v = 1.7156 * (v - 0.5)

				! Evaluate the quadratic form
				x = u - s
				y = ABS(v) - t
				q = x**2 + y*(a*y - b*x)

				! Accept/reject
				if(q < r1) exit
				if(q > r2) cycle
				if(v**2 < -4.0*LOG(u)*u**2) exit
			end do

			! Return ratio of P's coordinates as the normal deviate
			rvNormValues(i) = v/u
		
		end do

	end function Norm
	
	
	! This is all mine (Patti)
	function MultiNorm(rU, rV, rW, rUU, rVV, rWW, rUV, rUW, rVW, rvNormU, rvNormV, rvNormW) result(iRetCode)
	
		! Routine arguments
		real, intent(in)				:: rU
		real, intent(in)				:: rV
		real, intent(in)				:: rW
		real, intent(in)				:: rUU
		real, intent(in)				:: rVV
		real, intent(in)				:: rWW
		real, intent(in)				:: rUV
		real, intent(in)				:: rUW
		real, intent(in)				:: rVW
		real, dimension(:), intent(out)	:: rvNormU
		real, dimension(:), intent(out)	:: rvNormV
		real, dimension(:), intent(out)	:: rvNormW
		integer							:: iRetCode
		
		! Locals
		

		! Compute the coefficients of the Cholesky decomposition of covariances matrix
		L11 = SQRT(rUU)
		L12 = rUV/L11
		L13 = rUW/L11
		L22 = SQRT(-(rUV**2/rUU) + rVV)
		L23 = (-rUV*rUW + rUU*rVW) / &
				SQRT(rUU*(-rUV**2 + rUU*rVV))
		L33 = SQRT(-(rUW**2/rUU) - &
				(-rUV*rUW + rUU*rVW) * &
				(-rUV*rUW + rUU*rVW) / &
				(rUU**2*(-(rUV**2/rUU) + rVV)) + rWW)
		
		! Update particles positions
		DO i = 1, iNumPart
		
			! Generate three random values distributed according to the multivariate
			! normal distribution with same covariance matrix as computed
			rRandX = random_Normal()
			rRandY = random_Normal()
			rRandZ = random_Normal()
			rRandU = L11*rRandX + L12*rRandY + L13*rRandZ
			rRandV =              L22*rRandY + L23*rRandZ
			rRandW =                           L33*rRandZ

end module NormalDeviates
