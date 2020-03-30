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
	public	:: random_normal

contains

	function random_normal() result(rNorm)
	
		! Routine arguments
		real	:: rNorm

		! Locals
		real	:: u, v, x, y, q
		
		! Constants
		real, parameter	:: s  =  0.449871
		real, parameter	:: t  = -0.386595
		real, parameter	:: a  =  0.19600
		real, parameter	:: b  =  0.25472
		real, parameter	:: r1 =  0.27597
		real, parameter	:: r2 =  0.27846

	!     Generate P = (u,v) uniform in rectangle enclosing acceptance region

	DO
	  CALL RANDOM_NUMBER(u)
	  CALL RANDOM_NUMBER(v)
	  v = 1.7156 * (v - 0.5)

	!     Evaluate the quadratic form
	  x = u - s
	  y = ABS(v) - t
	  q = x**2 + y*(a*y - b*x)

	!     Accept P if inside inner ellipse
	  IF (q < r1) EXIT
	!     Reject P if outside outer ellipse
	  IF (q > r2) CYCLE
	!     Reject P if outside acceptance region
	  IF (v**2 < -4.0*LOG(u)*u**2) EXIT
	END DO

	!     Return ratio of P's coordinates as the normal deviate
	random_normal = v/u
	RETURN

	end function random_normal

end module NormalDeviates
