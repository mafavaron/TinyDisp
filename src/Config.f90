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
	
	type IniFile
		logical, private										:: lIsUseable
		integer, private										:: iNumKeys
		character(len=256), dimension(:), allocatable, private	:: svLine
		character(len=256), dimension(:), allocatable, private	:: svKey
		character(len=256), dimension(:), allocatable, private	:: svValue
	contains
		! Constructor
		procedure, public	:: read       => iniRead
		procedure, public	:: dump       => iniDump
		procedure, public	:: getString  => iniGetString
		procedure, public	:: getReal4   => iniGetReal4
		procedure, public	:: getReal8   => iniGetReal8
		procedure, public	:: getInteger => iniGetInteger
	end type IniFile

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
