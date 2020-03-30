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
	public	:: MultiNorm

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
		integer							:: n
		integer							:: iErrCode
		real							:: rL11
		real							:: rL12
		real							:: rL13
		real							:: rL22
		real							:: rL23
		real							:: rL33
		real, dimension(:), allocatable	:: rvNormX
		real, dimension(:), allocatable	:: rvNormY
		real, dimension(:), allocatable	:: rvNormZ
		
		! Assume success (will falsify on failure)
		iRetCode = 0
		
		! Check input parameters
		if(rUU < 0. .or. rVV < 0. .or. rWW < 0.) then
			iRetCode = 1
			return
		end if
		n = size(rvNormU)
		if(n <= 0) then
			iRetCode = 2
			return
		end if
		if(size(rvNormV) /= n .or. size(rvNormW) /= n) then
			iRetCode = 2
			return
		end if

		! Compute the coefficients of the Cholesky decomposition of covariances matrix
		rL11 = SQRT(rUU)
		rL12 = rUV/L11
		rL13 = rUW/L11
		rL22 = SQRT(-(rUV**2/rUU) + rVV)
		rL23 = (-rUV*rUW + rUU*rVW) / SQRT(rUU*(-rUV**2 + rUU*rVV))
		rL33 = SQRT(-(rUW**2/rUU) - (-rUV*rUW + rUU*rVW) * (-rUV*rUW + rUU*rVW) / (rUU**2*(-(rUV**2/rUU) + rVV)) + rWW)

		! Generate three random values distributed according to the multivariate
		! normal distribution with same covariance matrix as computed
		allocate(rvNormX(n), rvNormY(n), rvNormZ(n))
		iErrCode = Norm(rvNormX)
		if(iErrCode /= 0) then
			iRetCode = 3
			deallocate(rvNormX, rvNormY, rvNormZ)
			return
		end if
		iErrCode = Norm(rvNormY)
		if(iErrCode /= 0) then
			iRetCode = 3
			deallocate(rvNormX, rvNormY, rvNormZ)
			return
		end if
		iErrCode = Norm(rvNormZ)
		if(iErrCode /= 0) then
			iRetCode = 3
			deallocate(rvNormX, rvNormY, rvNormZ)
			return
		end if
		rvNormU = rL11*rvNormX + rL12*rvNormY + rL13*rvNormZ + rU
		rvNormV =                rL22*rvNormY + rL23*rvNormZ + rV
		rvNormW =                               rL33*rvNormZ + rW
		
		! Leave
		deallocate(rvNormX, rvNormY, rvNormZ)
		
	end function MultiNorm

end module NormalDeviates
