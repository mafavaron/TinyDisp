! options - Fortran module for options treatment in artificial case generator.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module options

    use calendar

    implicit none
    
    private
    
    ! Public interface
    public  :: printUsage
    public  :: iNumFrames
    public  :: iTimeStep
    public  :: iStartTime
    public  :: sOutputFile
    public  :: rVel
    public  :: rDir
    
    ! Internal state
    integer             :: iNumFrames
    integer             :: iTimeStep
    integer             :: iStartTime
    character(len=256)  :: sOutputFile
    real                :: rVel
    real                :: rDir
    
contains

    subroutine printUsage
    
        ! Routine arguments
        ! --none--
        
        ! Locals
        ! --none--
        
        ! Print explanatory message
        print *, 'artificial - Case generator for TinyDisp'
        print *
        print *, 'Usage:'
        print *
        print *, '  artificial <num_frames> <time_step> <start_datetime> <out_file> <option> ...parameters...'
        print *
        print *, 'where <option> may assume the following values:'
        print *
        print *, '--constant <wind_speed> <wind_provenance_direction>'
        print *
        print *, ''
        print *
        print *
        print *
        print *
        print *
        print *
        print *
        print *
        print *
        print *
        
    end subroutine printUsage
    
    
    function decodeOptions() result(iOptCode)
    
        ! Routine arguments
        integer :: iOptCode
        
        ! Locals
        integer             :: iYear, iMonth, iDay, iHour, iMinute, iSecond
        character(len=256)  :: sOption
        character(len=256)  :: sBuffer
        integer             :: iNumParameter
        integer             :: iErrCode
        
        ! Preliminary check: is the mandatory, common part here?
        iNumParameter = command_argument_count()
        if(iNumParameter < 5) then
            iOptCode = -1
            return
        end if
        
        ! Get the common part, and encode it into state variables
        
        call get_command_argument(1, sBuffer)
        read(sBuffer, *, iostat = iErrCode) iNumFrames
        if(iErrCode /= 0) then
            iOptCode = -2
            return
        end if
        
        call get_command_argument(2, sBuffer)
        read(sBuffer, *, iostat = iErrCode) iTimeStep
        if(iErrCode /= 0) then
            iOptCode = -3
            return
        end if
        
        call get_command_argument(3, sBuffer)
        read(sBuffer, "(i4.4,5(1x,i2.2))", iostat = iErrCode) iYear, iMonth, iDay, iHour, iMinute, iSecond
        if(iErrCode /= 0) then
            iOptCode = -4
            return
        end if
        
        call get_command_argument(4, sOutputFile)
        
        call get_command_argument(4, sOption)
        
    end function decodeOptions
    
    ! *********************
    ! * Internal routines *
    ! *********************
    
    subroutine toLower(sString)
    
        ! Routine arguments
        character(len=*), intent(inout) :: sString
        
        ! Locals
        integer :: i
        
        ! Change case to lower in place
        do i = 1, len_trim(sString)
            if(sString(i:i) >= 'A' .and. sString(i:i) <= 'Z') then
                sString(i:i) = char(ichar(sString(i:i)) - ichar('A') + ichar('a'))
            end if
        end do
        
    end subroutine toLower

end module options
