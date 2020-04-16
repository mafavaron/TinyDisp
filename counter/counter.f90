! counter - Program to summarize gridded particles counts from TinyDisp.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program counter

    use Calendar
    
    implicit none
    
    ! Locals
    character(len=256)                      :: sInputFile
    character(len=256)                      :: sOutputFile
    integer, dimension(:,:), allocatable    :: imCount
    integer, dimension(:,:), allocatable    :: imTotal
    integer                                 :: iYear, iMonth, iDay, iHour, iMinute, iSecond
    
    ! Get parameters
    if(command_argument_count() /= 2) then
        print *, "counter - Program to summarize gridded particles counts from TinyDisp"
        print *
        print *, "Usage:"
        print *
        print *, "  counter <Gridded_Count_File> <Output_File>"
        print *
        print *, "Copyright 2020 by Servizi Territorio srl"
        print *, "                  This is open-source software, covered by the MIT license"
        print *
        stop
    end if
    call getarg(1, sInputFile)
    call getarg(2, sOutputFile)

end program counter
