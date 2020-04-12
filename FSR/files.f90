! files.f90 - Module, incorporating directory scans and access to FastSonic files
!
! Copyright 2019 by Servizi Territorio srl
!                   This is open-source code, covered by the MIT license
!
! Written by: Patti Favaron
!
module files

    implicit none

    private

    ! Public interface
    public  :: fileRead

contains

    function fileRead(sFileName, rvTimeStamp, rvU, rvV, rvW, rvT, rmQuantity, svQuantity) result(iRetCode)

        ! Routine arguments
        real(4), dimension(:), allocatable, intent(inout)       :: rvTimeStamp
        real(4), dimension(:), allocatable, intent(inout)       :: rvU
        real(4), dimension(:), allocatable, intent(inout)       :: rvV
        real(4), dimension(:), allocatable, intent(inout)       :: rvW
        real(4), dimension(:), allocatable, intent(inout)       :: rvT
        real(4), dimension(:,:), allocatable, intent(inout)     :: rmQuantity
        character(8), dimension(:), allocatable, intent(inout)  :: svQuantity
        character(len=*), intent(in)                            :: sFileName
        integer                                                 :: iRetCode

        ! Locals
        integer     :: iLUN
        integer     :: iErrCode
        integer     :: iNumData
        integer(2)  :: iNumQuantities
        integer     :: iQuantity

        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Try accessing file
        open(newunit=iLUN, file=sFileName, status='old', action='read', access='stream', iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if

        ! Get number of data and use it to reserve workspace
        read(iLUN, iostat=iErrCode) iNumData
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 2
            return
        end if
        if(iNumData <= 0) then
            close(iLUN)
            iRetCode = 3
            return
        end if
        read(iLUN, iostat=iErrCode) iNumQuantities
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 4
            return
        end if
        iErrCode = fileClean(rvTimeStamp, rvU, rvV, rvW, rvT, rmQuantity, svQuantity)
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 5
            return
        end if
        allocate(rvTimeStamp(iNumData))
        allocate(rvU(iNumData))
        allocate(rvV(iNumData))
        allocate(rvW(iNumData))
        allocate(rvT(iNumData))
        allocate(rmQuantity(iNumData, iNumQuantities))
        allocate(svQuantity(iNumQuantities))

        ! Gather quantity names
        do iQuantity = 1, iNumQuantities
            read(iLUN, iostat=iErrCode) svQuantity(iQuantity)
            if(iErrCode /= 0) then
                close(iLUN)
                iRetCode = 6
                return
            end if
        end do

        ! Get actual data
        read(iLUN, iostat=iErrCode) rvTimeStamp
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 7
            return
        end if
        read(iLUN, iostat=iErrCode) rvU
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 8
            return
        end if
        read(iLUN, iostat=iErrCode) rvV
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 9
            return
        end if
        read(iLUN, iostat=iErrCode) rvW
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 10
            return
        end if
        read(iLUN, iostat=iErrCode) rvT
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 11
            return
        end if
        read(iLUN, iostat=iErrCode) rmQuantity
        if(iErrCode /= 0) then
            close(iLUN)
            iRetCode = 12
            return
        end if

        ! Leave
        close(iLUN)

    end function fileRead


    function fileClean(rvTimeStamp, rvU, rvV, rvW, rvT, rmQuantity, svQuantity) result(iRetCode)

        ! Routine arguments
        real(4), dimension(:), allocatable, intent(inout)       :: rvTimeStamp
        real(4), dimension(:), allocatable, intent(inout)       :: rvU
        real(4), dimension(:), allocatable, intent(inout)       :: rvV
        real(4), dimension(:), allocatable, intent(inout)       :: rvW
        real(4), dimension(:), allocatable, intent(inout)       :: rvT
        real(4), dimension(:,:), allocatable, intent(inout)     :: rmQuantity
        character(8), dimension(:), allocatable, intent(inout)  :: svQuantity
        integer                                                 :: iRetCode

        ! Locals
        ! --none--

        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Release workspace, if any
        if(allocated(rvTimeStamp)) deallocate(rvTimeStamp)
        if(allocated(rvU))         deallocate(rvU)
        if(allocated(rvV))         deallocate(rvV)
        if(allocated(rvW))         deallocate(rvW)
        if(allocated(rvT))         deallocate(rvT)
        if(allocated(rmQuantity))  deallocate(rmQuantity)
        if(allocated(svQuantity))  deallocate(svQuantity)

    end function fileClean

end module files
