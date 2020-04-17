! artificial - Artificial case generator for TinyDisp.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program artificial

    use calendar
    use options
    
    implicit none
    
    ! Locals
    integer :: iOptCode
    integer :: iFrame
    integer :: iCurTime
    integer :: iYear, iMonth, iDay, iHour, iMinute, iSecond
    
    ! Constants (please don't change)
    real, parameter :: TO_RAD = 3.1415926535 / 180.0
    
    ! Decode command line
    iOptCode = decodeOptions()
    if(iOptCode <= 0) then
        print *, "artificial:: error: Some command line parameter is wrong - Opt.code = ", iOptCode
        stop
    end if
    
    ! Execute the selected command
    
    if(iOptCode == 1) then  ! --constant
    
        open(10, file=sOutputFile, status='unknown', action='write')
        write(10, "('Time.Stamp, U, V, W, StdDev.U, StdDev.V, StdDev.W, Cov.UV, Cov.UW, Cov.VW')")
        do iFrame = 1, iNumFrames
            iCurTime = iStartTime + (iFrame - 1) * iTimeStep
            call unpacktime(iCurTime, iYear, iMonth, iDay, iHour, iMinute, iSecond)
            write(10, "(i4.4,2('-',i2.2),1x,i2.2,2(':',i2.2),3(',',f8.2),6(',',f8.4))") &
                iYear, iMonth, iDay, iHour, iMinute, iSecond, &
                rVel * cos(TO_RAD * rDir), rVel * sin(TO_RAD * rDir), &
                0.0, &
                rSigma, rSigma, &
                0.0, &
                rCov, &
                0.0, 0.0
        end do
        close(10)
    
    elseif(iOptCode == 2) then
    
    else
    
        print *, "artificial:: error: Unsupported command"
        stop
    
    end if
    
end program artificial
