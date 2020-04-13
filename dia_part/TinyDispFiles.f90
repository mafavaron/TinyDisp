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
        ! Generalities
        logical                             :: lTwoDimensional
        integer                             :: iLUN
        integer                             :: iNumTimeSteps
        ! Meteorology
        integer                             :: iCurTime
        real                                :: rU
        real                                :: rV
        real                                :: rW
        real                                :: rStdDevU
        real                                :: rStdDevV
        real                                :: rStdDevW
        real                                :: rCovUV
        real                                :: rCovUW
        real                                :: rCovVW
        ! Particles
        integer, dimension(:), allocatable  :: ivTimeStamp
        real, dimension(:), allocatable     :: rvX
        real, dimension(:), allocatable     :: rvY
        real, dimension(:), allocatable     :: rvZ
    contains
        procedure :: connect    => prtOpen
        procedure :: get        => prtRead
        procedure :: disconnect => prtClose
    end type ParticlesFileType
    
contains

    function prtOpen(this, sInputFile) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(out)   :: this
        character(len=256), intent(in)          :: sInputFile
        integer                                 :: iRetCode
        
        ! Locals
        integer :: iLUN
        integer :: iErrCode
        integer :: iMaxPart
        integer :: iNumMeteo
        
        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Initialize structure
        this % lTwoDimensional = .false.
        
        ! Access file
        open(newunit=iLUN, file=sInputFile, status='old', action='read', access='stream', iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        
        ! Get heading information
        read(iLUN, iostat=iErrCode) iMaxPart, iNumMeteo
        if(iErrCode /= 0) then
            iRetCode = 2
            close(iLUN)
            return
        end if
        if(iMaxPart == 0 .or. iNumMeteo <= 0) then
            iRetCode = 3
            close(iLUN)
            return
        end if
        
        ! Store intermediates to structure, for any later use
        this % iLUN = iLUN
        if(iMaxPart < 0) then
            iMaxPart = -iMaxPart
        else
            this % lTwoDimensional = .true.
        end if
        this % iNumTimeSteps = iNumMeteo
        
    end function prtOpen


    function prtRead(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(inout) :: this
        integer                                 :: iRetCode
        
        ! Locals
        integer :: iLUN
        integer :: iErrCode
        integer :: iNumPart
        integer :: iIteration
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Get steering data
        iLUN = this % iLUN
        
        ! Get this block meteo data and size
        if(this % lTwoDimensional) then
            read(iLUN, iostat=iErrCode) &
                iIteration, &
                this % iCurTime, &
                this % rU, &
                this % rV, &
                this % rStdDevU, &
                this % rStdDevV, &
                this % rCovUV, &
                iNumPart
            this % rW       = 0.
            this % rStdDevW = 0.
            this % rCovUW   = 0.
            this % rCovVW   = 0.
        else
            read(iLUN, iostat=iErrCode) &
                iIteration, &
                this % iCurTime, &
                this % rU, &
                this % rV, &
                this % rW, &
                this % rStdDevU, &
                this % rStdDevV, &
                this % rStdDevW, &
                this % rCovUV, &
                this % rCovUW, &
                this % rCovVW, &
                iNumPart
        end if
        if(iErrCode /= 0) then
            iRetCode = -1
            return
        end if
        
        ! Reserve storage space, if needed
        if(allocated(this % ivTimeStamp)) deallocate(this % ivTimeStamp)
        if(allocated(this % rvX))         deallocate(this % rvX)
        if(allocated(this % rvY))         deallocate(this % rvY)
        if(allocated(this % rvZ))         deallocate(this % rvZ)
        allocate(this % ivTimeStamp(iNumPart))
        allocate(this % rvX(iNumPart))
        allocate(this % rvY(iNumPart))
        allocate(this % rvZ(iNumPart))
        
        ! Gather data
        if(this % lTwoDimensional) then
            read(iLUN, iostat=iErrCode) &
                this % rvX, &
                this % rvY, &
                this % ivTimeStamp
        else
            read(iLUN, iostat=iErrCode) &
                this % rvX, &
                this % rvY, &
                this % rvZ, &
                this % ivTimeStamp
        end if
        
    end function prtRead


    function prtClose(this) result(iRetCode)
    
        ! Routine arguments
        class(ParticlesFileType), intent(inout) :: this
        integer                                 :: iRetCode
        
        ! Locals
        integer :: iErrCode
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Disconnect file (doing nothing in case it is already disconnected)
        close(this % iLUN, iostat=iErrCode)
        
    end function prtClose

end module TinyDispFiles
