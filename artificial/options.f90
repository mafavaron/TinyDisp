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
    ! -1- Procedures
    public  :: printUsage
    public  :: decodeOptions
    ! -1- State
    public  :: iNumFrames
    public  :: iTimeStep
    public  :: iStartTime
    public  :: sOutputFile
    public  :: rVel
    public  :: rDir
    public  :: rSigma
    public  :: rCov
    public  :: iNumLoops
    
    ! Internal state
    integer             :: iNumFrames
    integer             :: iTimeStep
    integer             :: iStartTime
    character(len=256)  :: sOutputFile
    real                :: rVel
    real                :: rDir
    real                :: rSigma
    real                :: rCov
    integer             :: iNumLoops
    
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
        print *, '--constant <wind_speed> <wind_provenance_direction> <sigma> <covar>'
        print *
        print *, '--circular <wind_speed> <wind_provenance_direction> <sigma> <covar> <total_loops>'
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
        call packtime(iStartTime, iYear, iMonth, iDay, iHour, iMinute, iSecond)
        
        call get_command_argument(4, sOutputFile)
        
        call get_command_argument(5, sOption)
        call toLower(sOption)
        
        ! Options-specific processing
        if(sOption == "--constant") then
            
            if(iNumParameter /= 9) then
                iOptCode = -5
                return
            end if
            
            call get_command_argument(6, sBuffer)
            read(sBuffer, *, iostat = iErrCode) rVel
            if(iErrCode /= 0) then
                iOptCode = -6
                return
            end if
            
            call get_command_argument(7, sBuffer)
            read(sBuffer, *, iostat = iErrCode) rDir
            if(iErrCode /= 0) then
                iOptCode = -7
                return
            end if
            
            call get_command_argument(8, sBuffer)
            read(sBuffer, *, iostat = iErrCode) rSigma
            if(iErrCode /= 0) then
                iOptCode = -8
                return
            end if
            
            call get_command_argument(9, sBuffer)
            read(sBuffer, *, iostat = iErrCode) rCov
            if(iErrCode /= 0) then
                iOptCode = -9
                return
            end if
            
            iOptCode = 1
            
        elseif(sOption == "--circular") then
        
            if(iNumParameter /= 10) then
                iOptCode = -10
                return
            end if
            
            call get_command_argument(6, sBuffer)
            read(sBuffer, *, iostat = iErrCode) rVel
            if(iErrCode /= 0) then
                iOptCode = -11
                return
            end if
            
            call get_command_argument(7, sBuffer)
            read(sBuffer, *, iostat = iErrCode) rDir
            if(iErrCode /= 0) then
                iOptCode = -12
                return
            end if
            
            call get_command_argument(8, sBuffer)
            read(sBuffer, *, iostat = iErrCode) rSigma
            if(iErrCode /= 0) then
                iOptCode = -13
                return
            end if
            
            call get_command_argument(9, sBuffer)
            read(sBuffer, *, iostat = iErrCode) rCov
            if(iErrCode /= 0) then
                iOptCode = -14
                return
            end if
            
            call get_command_argument(10, sBuffer)
            read(sBuffer, *, iostat = iErrCode) iNumLoops
            if(iErrCode /= 0) then
                iOptCode = -15
                return
            end if
            
            iOptCode = 2
            
        end if
        
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
