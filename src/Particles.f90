! Particles - Fortran module for particles generation and dynamics.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module Particles

    use NormalDeviates
    
    implicit none
    
    private
    
    ! Public interface
    public  :: ParticlesPoolType
    
    ! Data types
    type ParticlesPoolType
        integer                             :: iNextPart
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        real, dimension(:), allocatable     :: rvZ
        real, dimension(:), allocatable     :: rvU
        real, dimension(:), allocatable     :: rvV
        real, dimension(:), allocatable     :: rvW
        integer, dimension(:), allocatable  :: ivTimeStampAtBirth
    end type ParticlesPoolType

end module Particles
