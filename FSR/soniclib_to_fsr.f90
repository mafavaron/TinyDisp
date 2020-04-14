! Main program: iterate over SonicLib files, get their sonic part,
!               and save it in new FastSonic form.
!
! By: Patti Favaron
!
program Soniclib_To_FSR

    use fileList

    implicit none

    ! Locals
    character(len=256)  :: sInputListFile
    character(len=256)  :: sOutputPath
    integer             :: iRetCode
    character(len=256)  :: sInputFileName
    character(len=256)  :: sOutputFileName
    character(len=256)  :: sBuffer
    integer             :: i
    integer             :: iNumData
    integer             :: iData
    integer             :: iTimeStamp
    real                :: rU, rV, rW, rT
    integer(2)          :: zero = 0
    real, dimension(:), allocatable                 :: rvTimeStamp, rvU, rvV, rvW, rvT
    character(len=256), dimension(:), allocatable   :: svFiles

    ! Get command arguments
    if(command_argument_count() /= 2) then
        print *, "soniclib_to_fsr - Ultrasonic anemometer raw data encoding procedure"
        print *
        print *, "error:: Invalid command line"
        print *
        print *, "Usage:"
        print *
        print *, "  soniclib_to_fsr <Soniclib_Files_List> <FSR_Path>"
        print *
        print *, "Copyright 2020 by Servizi Territorio srl"
        print *, "                  This software is open source, covered by the MIT license"
        print *
        stop
    end if
    call get_command_argument(1, sInputListFile)
    call get_command_argument(2, sOutputPath)

    ! Identify files in input path
    iRetCode = readFileList(sInputListFile, svFiles)
    if(iRetCode /= 0) then
        print *, "soniclib_to_fsr:: error: List of input files not read - Return code = ", iRetCode
        stop
    end if
    
    ! Iterate over files
    do i = 1, size(svFiles)
    
        sInputFileName = svFiles(i)
        sOutputFileName = trim(sOutputPath) // '\\' // baseName(sInputFileName, '\\')
        sOutputFileName = sOutputFileName(1:len_trim(sOutputFileName)-3) // 'fsr'

        ! Process file
        ! -1- Count lines
        print *, "Encoding to ", trim(sOutputFileName)
        open(10, file=sInputFileName, status='old', action='read', iostat=iRetCode)
        if(iRetCode /= 0) then
            print *,trim(sInputFileName)
            print *, 'error:: Input file not opened - ', iRetCode
            stop
        end if
        ! -1- Count data in file
        read(10, "(a)", iostat=iRetCode) sBuffer
        if(iRetCode /= 0) then
            print *, 'error:: Empty input file'
            cycle
        end if
        iNumData = 0
        do
            read(10, "(a)", iostat=iRetCode) sBuffer
            if(iRetCode /= 0) exit
            iNumData = iNumData + 1
        end do
        if(iNumData <= 0) then
            print *, 'error:: No data in input file'
            cycle
        end if
        ! -1- Reserve workspace
        if(allocated(rvTimeStamp)) deallocate(rvTimeStamp)
        allocate(rvTimeStamp(iNumData))
        if(allocated(rvU)) deallocate(rvU)
        allocate(rvU(iNumData))
        if(allocated(rvV)) deallocate(rvV)
        allocate(rvV(iNumData))
        if(allocated(rvW)) deallocate(rvW)
        allocate(rvW(iNumData))
        if(allocated(rvT)) deallocate(rvT)
        allocate(rvT(iNumData))
        ! -1- Really read data
        rewind(10)
        read(10, "(a)") sBuffer
        do iData = 1, iNumData
            read(10, *) iTimeStamp, rU, rV, rW, rT
            rvTimeStamp(iData) = iTimeStamp
            rvU(iData) = rU
            rvV(iData) = rV
            rvW(iData) = rW
            rvT(iData) = rT
        end do
        close(10)

        ! Write data in binary form
        open(11, file=sOutputFileName, status='unknown', action='write', access='stream')
        write(11) iNumData
        write(11) zero
        write(11) rvTimeStamp
        write(11) rvU
        write(11) rvV
        write(11) rvW
        write(11) rvT
        close(11)

    end do

    ! Time elapsed counts
    print *, "*** END JOB ***"

end program Soniclib_To_FSR
