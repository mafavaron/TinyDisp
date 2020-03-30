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
	public	:: rNorm

contains

	! Credit: this routine has been obtained by refactoring of function "random_normal"
	! by Alan Miller (CSIRO Division of Mathematics & Statistics), module 'random.f90'
	elemental function rNorm() result(rNormValue)
	
		! Routine arguments
		real	:: rNormValue

		! Locals
		real	:: u, v, x, y, q
		
		! Constants
		real, parameter	:: s  =  0.449871
		real, parameter	:: t  = -0.386595
		real, parameter	:: a  =  0.19600
		real, parameter	:: b  =  0.25472
		real, parameter	:: r1 =  0.27597
		real, parameter	:: r2 =  0.27846

		! Main loop: generate P = (u,v) uniform in rectangle enclosing
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
		rNormValue = v/u

	end function rNorm

end module NormalDeviates
