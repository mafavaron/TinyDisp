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
        
    end function decodeOptions

end module options
