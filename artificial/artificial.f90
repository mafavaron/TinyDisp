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
    
    ! Decode command line
    iOptCode = decodeOptions()
    if(iOptCode <= 0) then
        print *, "artificial:: error: Some command line parameter is wrong - Ret.code = ", iOptCode
    end if
    
end program artificial
