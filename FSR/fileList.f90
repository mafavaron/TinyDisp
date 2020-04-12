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
        integer             :: iErrCode
        integer             :: iNumFiles
        integer             :: iFile
        character(len=256)  :: sBuffer
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! First step: Count lines in input file
        iNumFiles = 0
        open(iLUN, file=sFileName, status='old', action='read', iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        do
            read(iLUN, "(a)", iostat=iErrCode) sBuffer
            if(iErrCode /= 0) exit
            iNumFiles = iNumFiles + 1
        end do
        if(iNumFiles <= 0) then
            iRetCode = 2
            close(iLUN)
            return
        end if
        if(allocated(svFiles)) deallocate(svFiles)
        allocate(svFiles(iNumFiles))
        
        ! Second step: Actual read
        rewind(iLUN)
        do iFile = 1, iNumFiles
            read(iLUN, "(a)") svFiles(iFile)
        end do
        close(iLUN)
        
    end function readFileList

end module fileList
