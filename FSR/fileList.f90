! Module fileList, fr gathering and making available the data files list.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module fileList

    implicit none
    
    private
    
    ! Public interface
    public  :: svFiles
    public  :: readFileList
    
    ! Declarations
    
    
contains

    function readFileList(iLUN, sFileName, svFiles) result(iRetCode)
    
        ! Routine arguments
        integer, intent(in)                                         :: iLUN
        character(len=256), intent(in)                              :: sFileName
        character(len=256), dimension(:), allocatable, intent(out)  :: svFiles
        integer                                                     :: iRetCode
        
        ! Locals
        integer :: iErrCode
        integer :: iNumFiles
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
    end function readFileList

end module fileList
