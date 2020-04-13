! Module TinyDispFiles
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module TinyDispFiles

    implicit none
    
    private
    
    ! Public interface
    public  :: ParticlesFileType
    
    ! Data types
    
    type ParticlesFileType
        logical                             :: lTwoDimensional
        integer, dimension(:), allocatable  :: ivTimeStamp
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        real, dimension(:), allocatable     :: rvZ
    contains
        procedure open
        procedure read
    end type ParticlesFileType
    
contains

    function open(this) result(iRetCode)
    end function open

    function read(this) result(iRetCode)
    end function read

end module TinyDispFiles
