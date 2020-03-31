! NormalDeviates - Fortran module for generating univariate and three-variate
!                  random deviates.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module Config

	implicit none
	
	private
	
	! Public interface
	public	:: ConfigType
	
	! Data types
	
	type ConfigType
		! General
		logical				:: lTwoDimensionalRun
		! Domain
		real				:: rEdgeLength
		! Particles
		integer				:: iNumPartsEmittedPerStep
		integer				:: iTimeStep
		! Counting grid
		integer				:: iNumCells
		real				:: rXmin
		real				:: rYmin
		real				:: rXmax
		real				:: rYmax
	contains
		procedure			:: get
	end type ConfigType
	
contains

	function get(this, iLUN, sFileName) result(iRetCode)
	
		! Routine arguments
		class(ConfigType), intent(out)	:: this
		integer, intent(in)				:: iLUN
		character(len=*), intent(in)	:: sFileName
		integer							:: iRetCode
		
		! Locals
		
		! Assume success (will falsify on failure)
		iRetCode = 0
		
	end function get

end module Config
