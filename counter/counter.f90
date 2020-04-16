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
    integer                                 :: iRetCode
    character(len=256)                      :: sInputFile
    character(len=256)                      :: sOutputFile
    integer, dimension(:,:), allocatable    :: imCount
    integer, dimension(:,:), allocatable    :: imTotal
    integer                                 :: iTimeStamp
    integer                                 :: iYear, iMonth, iDay, iHour, iMinute, iSecond
    real                                    :: rXmin
    real                                    :: rYmin
    real                                    :: rDxy
    integer                                 :: iNumCells
    
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
    
    ! Access the count file
    open(10, file=sInputFile, action="read", status="old", access='stream', iostat=iRetCode)
    if(iRetCode /= 0) then
        print *, 'counter:: error: Input file not opened'
        stop
    end if
    read(10, iostat=iRetCode) rXmin, rYmin, rDxy, iNumCells
    if(iRetCode /= 0) then
        print *, 'counter:: error: Input file is invalid'
        stop
    end if
    allocate(imCount(iNumCells,iNumCells), imTotal(iNumCells,iNumCells))
    
    ! Main loop: iterate over count matrices
    imTotal = 0
    do
        read(10, iostat=iRetCode) iTimeStamp
        if(iRetCode /= 0) exit
        call unpacktime(iTimeStamp, iYear, iMonth, iDay, iHour, iMinute, iSecond)
        read(10) imCount
        imTotal = imTotal + imCount
        print "(i4.4,2('-',i2.2),1x,i2.2,2(':',i2.2), 1x, i10)", &
            iYear, iMonth, iDay, iHour, iMinute, iSecond, sum(imCount)
    end do
    
    ! Leave
    deallocate(imCount, imTotal)
    close(10)

end program counter
