! Config - Fortran module for retrieving and distributing configuration data.
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
module Config

    implicit none
    
    private
    
    ! Public interface
    public    :: ConfigType
    
    ! Data types
    
    type ConfigType
        ! General
        logical                 :: lValid
        integer                 :: iDebugLevel
        ! Domain
        real                    :: rEdgeLength
        real                    :: rXmin
        real                    :: rYmin
        real                    :: rXmax
        real                    :: rYmax
        ! Meteo file
        character(len=256)      :: sMeteoFile
        ! Particles
        integer                 :: iNumPartsEmittedPerStep
        integer                 :: iTimeStep
        real                    :: rInertia                 ! In [0,1]; 0: previous speed not considered; 1: previous speed considered
        integer                 :: iSurvivalTime            ! Proper multiple of iTimeStep
        integer                 :: iMaxPart
        ! Dump
        character(len=256)      :: sParticlesFile
        logical                 :: lTwoDimensional
        ! Grid
        logical                 :: lEnableCounting
        integer                 :: iNumCells
        real                    :: rDxy
        character(len=256)      :: sCountingFile
    contains
        procedure               :: get
    end type ConfigType
    
    type IniFile
        logical, private                                            :: lIsUseable
        integer, private                                            :: iNumKeys
        character(len=256), dimension(:), allocatable, private      :: svLine
        character(len=256), dimension(:), allocatable, private      :: svKey
        character(len=256), dimension(:), allocatable, private      :: svValue
    contains
        ! Constructor
        procedure, public    :: read       => iniRead
        procedure, public    :: dump       => iniDump
        procedure, public    :: getString  => iniGetString
        procedure, public    :: getReal4   => iniGetReal4
        procedure, public    :: getReal8   => iniGetReal8
        procedure, public    :: getInteger => iniGetInteger
    end type IniFile

contains

    function get(this, iLUN, sFileName) result(iRetCode)
    
        ! Routine arguments
        class(ConfigType), intent(out)      :: this
        integer, intent(in)                 :: iLUN
        character(len=*), intent(in)        :: sFileName
        integer                             :: iRetCode
        
        ! Locals
        type(IniFile)       :: tIni
        integer             :: iErrCode
        integer             :: iDimensions
        logical             :: lIsMeteo
        
        ! Assume success (will falsify on failure)
        iRetCode = 0
        
        ! Access configuration file
        iErrCode = tIni % read(iLUN, sFileName)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        
        ! General
        iErrCode = tIni % getInteger("Particles", "NumPartsEmittedPerStep", this % iDebugLevel, 0)
        if(iErrCode /= 0) then
            iRetCode = 2
            return
        end if
        if(this % iDebugLevel < 0) then
            iRetCode = 2
            return
        end if
        
        ! Domain
        iErrCode = tIni % getReal4("Domain", "EdgeLength", this % rEdgeLength, 0.)
        if(iErrCode /= 0) then
            iRetCode = 3
            return
        end if
        if(this % rEdgeLength <= 0.) then
            iRetCode = 3
            return
        end if
        
        ! Particles
        iErrCode = tIni % getInteger("Particles", "NumPartsEmittedPerStep", this % iNumPartsEmittedPerStep, 0)
        if(iErrCode /= 0) then
            iRetCode = 4
            return
        end if
        if(this % iNumPartsEmittedPerStep <= 0) then
            iRetCode = 4
            return
        end if
        iErrCode = tIni % getInteger("Particles", "TimeStep", this % iTimeStep, 0)
        if(iErrCode /= 0) then
            iRetCode = 5
            return
        end if
        if(this % iTimeStep <= 0) then
            iRetCode = 5
            return
        end if
        iErrCode = tIni % getReal4("Particles", "Inertia", this % rInertia, 0.)
        if(iErrCode /= 0) then
            iRetCode = 6
            return
        end if
        if(this % rInertia < 0. .or. this % rInertia > 1.) then
            iRetCode = 6
            return
        end if
        iErrCode = tIni % getInteger("Particles", "SurvivalTime", this % iSurvivalTime, 0)
        if(iErrCode /= 0) then
            iRetCode = 7
            return
        end if
        if(mod(this % iSurvivalTime, this % iTimeStep) /= 0) then
            iRetCode = 7
            return
        end if
        
        ! Dump
        iErrCode = tIni % getString("Dump", "ParticlesFile", this % sParticlesFile, "")
        if(iErrCode /= 0) then
            iRetCode = 8
            return
        end if
        iErrCode = tIni % getInteger("Dump", "Dimensions", iDimensions, 0)
        if(iErrCode /= 0) then
            iRetCode = 9
            return
        end if
        if(iDimensions /= 2 .and. iDimensions /= 3) then
            iRetCode = 9
            return
        end if
        this % lTwoDimensional = iDimensions == 2
        
        ! Grid
        iErrCode = tIni % getString("Grid", "CountingFile", this % sCountingFile, "")
        if(iErrCode /= 0) then
            iRetCode = 10
            return
        end if
        iErrCode = tIni % getInteger("Grid", "NumCells", this % iNumCells, 0)
        if(iErrCode /= 0) then
            iRetCode = 11
            return
        end if
        if(this % iNumCells <= 0) then
            iRetCode = 11
            return
        end if
        this % lEnableCounting = this % sCountingFile /= " "
        
        ! Meteo
        iErrCode = tIni % getString("Meteo", "MeteoFile", this % sMeteoFile, "")
        if(iErrCode /= 0) then
            iRetCode = 12
            return
        end if
        inquire(file=this % sMeteoFile, exist=lIsMeteo)
        if(.not.lIsMeteo) then
            iRetCode = 12
            return
        end if
        
        ! Compute derived quantities
        this % iMaxPart = this % iNumPartsEmittedPerStep * this % iSurvivalTime / this % iTimeStep
        if(this % lEnableCounting) then
            this % rDxy  =  this % rEdgeLength / this % iNumCells
            this % rXmin = -this % rEdgeLength / 2.
            this % rXmax = -this % rXmin
            this % rYmin =  this % rXmin
            this % rYmax =  this % rXmax
        else
            this % rDxy  = 0.
            this % rXmin = 0.
            this % rXmax = 0.
            this % rYmin = 0.
            this % rYmax = 0.
        end if
        
    end function get
    

    function iniRead(this, iLUN, sIniFileName) result(iRetCode)

        ! Routine arguments
        class(IniFile), intent(inout)       :: this
        integer, intent(in)                 :: iLUN
        character(len=*), intent(in)        :: sIniFileName
        integer                             :: iRetCode

        ! Locals
        integer                 :: iErrCode
        character(len=256)      :: sBuffer
        character(len=256)      :: sCurrentSection
        character(len=256)      :: sCurSection
        integer                 :: iNumLines
        integer                 :: iLine
        integer                 :: iPos
        integer                 :: iNumKeys
        integer                 :: i

        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Clean state before to proceed
        this % lIsUseable = .false.
        if(allocated(this % svLine)) deallocate(this % svLine)
        if(allocated(this % svKey)) deallocate(this % svKey)
        if(allocated(this % svValue)) deallocate(this % svValue)

        ! Now, count lines excluding comments
        open(iLUN, file=sIniFileName, status='old', action='read', iostat=iErrCode)
        if(iErrCode /= 0) then
            iRetCode = 1
            return
        end if
        iNumLines = 0
        do

            ! Try gathering a line, and if acquired replace all characters
            ! from the first '#' on with blanks
            read(iLun, "(a)", iostat=iErrCode) sBuffer
            if(iErrCode /= 0) exit
            iPos = index(sBuffer, "#")
            if(iPos > 0) sBuffer(iPos:) = ' '

            ! Replace TABs and other spaces with regular blanks
            do i = 1, len(sBuffer)
                if(ichar(sBuffer(i:i)) < 32) sBuffer(i:i) = ' '
            end do
            if(sBuffer == ' ') cycle
            ! Post-condition: something remains

            ! Increment line count, remembering lines which may be subject to parsing
            iNumLines = iNumLines + 1

        end do
        if(iNumLines <= 0) then
            close(iLun)
            iRetCode = 2
            return
        end if
        rewind(iLUN)

        ! Reserve workspace, and populate it with non-comment lines
        allocate(this % svLine(iNumLines), this % svKey(iNumLines), this % svValue(iNumLines))
        iLine = 0
        do

            ! Try gathering a line, and if acquired replace all characters
            ! from the first '#' on with blanks
            read(iLun, "(a)", iostat=iErrCode) sBuffer
            if(iErrCode /= 0) exit
            iPos = index(sBuffer, "#")
            if(iPos > 0) sBuffer(iPos:) = ' '

            ! Replace TABs and other spaces with regular blanks
            do i = 1, len(sBuffer)
                if(ichar(sBuffer(i:i)) < 32) sBuffer(i:i) = ' '
            end do
            if(sBuffer == ' ') cycle
            ! Post-condition: something remains

            ! Add next line
            iLine = iLine + 1
            this % svLine(iLine) = sBuffer

        end do
        close(iLUN)
        ! Post-condition: Some lines found

        ! Parse line contents
        sCurrentSection = ""
        iNumKeys        = 0
        do iLine = 1, iNumLines

            ! Check string is a section, and if so assign it
            if(isSection(this % svLine(iLine), sCurSection)) then
                sCurrentSection = sCurSection
            else
                ! Not a section: may contain an equal sign, that is, to be a name = value couple
                iPos = index(this % svLine(iLine), "=")
                if(iPos > 0) then
                    iNumKeys = iNumKeys + 1
                    write(this % svKey(iNumKeys), "(a,'@',a)") &
                        trim(sCurrentSection), adjustl(this % svLine(iLine)(1:(iPos-1)))
                    this % svValue(iNumKeys) = adjustl(this % svLine(iLine)((iPos+1):))
                    call removeChar(this % svValue(iNumKeys), '"')
                end if
            end if

        end do

        ! Confirm successful completion
        this % lIsUseable = .true.
        this % iNumKeys   = iNumKeys

    end function iniRead


    function iniDump(this) result(iRetCode)

        ! Routine arguments
        class(IniFile), intent(in)      :: this
        integer                         :: iRetCode

        ! Locals
        integer    :: i
        integer    :: iKeyLen

        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Check whether the dump is to be make in full,
        ! that is, the INI file read has completed successfully
        ! and the data structures have been filled
        if(this % lIsUseable) then

            ! Check length to constrain keys to when printing
            iKeyLen = 0
            do i = 1, this % iNumKeys
                iKeyLen = max(iKeyLen, len_trim(this % svKey(i)))
            end do

            ! Print all keys, and their associated values. To print
            ! keys in column the maximum key length is used, along with
            ! the fact that in Fortran all strings in an array share
            ! the same length and are blank-filled on right. The approach
            ! I've followed would have *not* worked in C and other
            ! variable-length string languages.
            do i = 1, this % iNumKeys
                print "(a,' -> ',a)", this % svKey(i)(1:iKeyLen), trim(this % svValue(i))
            end do

        else

            print *, "INI data contents has not yet been assigned, nothing to print"
            iRetCode = 1

        end if

    end function iniDump


    function iniGetString(this, sSection, sKey, sValue, sDefault) result(iRetCode)

        ! Routine arguments
        class(IniFile), intent(inout)               :: this
        character(len=*), intent(in)                :: sSection
        character(len=*), intent(in)                :: sKey
        character(len=*), intent(out)               :: sValue
        character(len=*), intent(in), optional      :: sDefault
        integer                                     :: iRetCode

        ! Locals
        integer                 :: i
        character(len=256)      :: sFullKey

        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Check something is to be made
        if(this % iNumKeys > 0) then

            ! Yes: there are data lines to scan
            write(sFullKey, "(a, '@', a)") trim(sSection), trim(sKey)
            do i = 1, this % iNumKeys
                if(this % svKey(i) == sFullKey) then
                    sValue = this % svValue(i)
                    return
                end if
            end do

            ! Nothing found if execution reaches here: in case,
            ! yield the default (if present) or a blank (otherwise).
            if(present(sDefault)) then
                sValue = sDefault
            else
                sValue = ' '
            end if

        else

            ! No INI data available: flag an error condition.
            if(present(sDefault)) then
                sValue = sDefault
            else
                sValue = ' '
            end if
            iRetCode = 1

        end if

    end function iniGetString


    function iniGetReal4(this, sSection, sKey, rValue, rDefault) result(iRetCode)

        ! Routine arguments
        class(IniFile), intent(inout)               :: this
        character(len=*), intent(in)                :: sSection
        character(len=*), intent(in)                :: sKey
        real, intent(out)                           :: rValue
        real, intent(in), optional                  :: rDefault
        integer                                     :: iRetCode

        ! Locals
        character(len=32)       :: sValue
        real                    :: rReplace
        integer                 :: iErrCode

        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Assign the replacement value based on rDefault
        if(present(rDefault)) then
            rReplace = rDefault
        else
            rReplace = -9999.9
        end if

        ! Gather the string supposedly containing the floating point value to transfer
        iErrCode = this % getString(sSection, sKey, sValue)
        if(iErrCode /= 0) then
            rValue = rReplace
            iRetCode = 1
            return
        end if
        ! Post-condition: iRetCode was 0 from now on

        ! Check the value found to be not empty
        if(sValue == ' ') then
            rValue = rReplace
            iRetCode = 2
            return
        end if

        ! Ok, something was found: but, it might not be a floating point value:
        ! try converting it and, in case of failure, yield an error
        read(sValue, *, iostat=iErrCode) rValue
        if(iErrCode /= 0) then
            rValue = rReplace
            iRetCode = 3
        end if
        ! Post-condition: 'rValue' has been assigned correctly, and on
        ! function exit will be restituted regularly

    end function iniGetReal4


    function iniGetReal8(this, sSection, sKey, rValue, rDefault) result(iRetCode)

        ! Routine arguments
        class(IniFile), intent(inout)               :: this
        character(len=*), intent(in)                :: sSection
        character(len=*), intent(in)                :: sKey
        real(8), intent(out)                        :: rValue
        real(8), intent(in), optional               :: rDefault
        integer                                     :: iRetCode

        ! Locals
        character(len=32)       :: sValue
        real(8)                 :: rReplace
        integer                 :: iErrCode

        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Assign the replacement value based on rDefault
        if(present(rDefault)) then
            rReplace = rDefault
        else
            rReplace = -9999.9d0
        end if

        ! Gather the string supposedly containing the floating point value to transfer
        iErrCode = this % getString(sSection, sKey, sValue)
        if(iErrCode /= 0) then
            rValue = rReplace
            iRetCode = 1
            return
        end if
        ! Post-condition: iRetCode was 0 from now on

        ! Check the value found to be not empty
        if(sValue == ' ') then
            rValue = rReplace
            iRetCode = 2
            return
        end if

        ! Ok, something was found: but, it might not be a floating point value:
        ! try converting it and, in case of failure, yield an error
        read(sValue, *, iostat=iErrCode) rValue
        if(iErrCode /= 0) then
            rValue = rReplace
            iRetCode = 3
        end if
        ! Post-condition: 'rValue' has been assigned correctly, and on
        ! function exit will be restituted regularly

    end function iniGetReal8


    function iniGetInteger(this, sSection, sKey, iValue, iDefault) result(iRetCode)

        ! Routine arguments
        class(IniFile), intent(inout)               :: this
        character(len=*), intent(in)                :: sSection
        character(len=*), intent(in)                :: sKey
        integer, intent(out)                        :: iValue
        integer, intent(in), optional               :: iDefault
        integer                                     :: iRetCode

        ! Locals
        character(len=32)       :: sValue
        integer                 :: iReplace
        integer                 :: iErrCode

        ! Assume success (will falsify on failure)
        iRetCode = 0

        ! Assign the replacement value based on rDefault
        if(present(iDefault)) then
            iReplace = iDefault
        else
            iReplace = -9999
        end if

        ! Gather the string supposedly containing the floating point value to transfer
        iErrCode = this % getString(sSection, sKey, sValue)
        if(iErrCode /= 0) then
            iValue = iReplace
            iRetCode = 1
            return
        end if
        ! Post-condition: iRetCode was 0 from now on

        ! Check the value found to be not empty
        if(sValue == ' ') then
            iValue = iReplace
            iRetCode = 2
            return
        end if

        ! Ok, something was found: but, it might not be a floating point value:
        ! try converting it and, in case of failure, yield an error
        read(sValue, *, iostat=iErrCode) iValue
        if(iErrCode /= 0) then
            iValue = iReplace
            iRetCode = 3
        end if
        ! Post-condition: 'iValue' has been assigned correctly, and on
        ! function exit will be restituted regularly

    end function iniGetInteger


	! **********************
	! * Internal functions *
	! **********************

    function isSection(sString, sSection) result(lIsSection)

        ! Routine arguments
        character(len=*), intent(in)    :: sString
        character(len=*), intent(out)   :: sSection
        logical                         :: lIsSection

		! Locals
        integer     :: iPos
        integer     :: iLast

		! Check first and last character are compatible with a section-type string
        iPos = verify(sString, ' ')
        iLast = len_trim(sString)
        if(iPos >= 1 .and. iPos <= iLast) then
            ! Some blanks before the string real beginning: parse from there
            lIsSection = sString(iPos:iPos) == '[' .and. sString(iLast:iLast) == ']'
            if(lIsSection) then
                sSection = sString((iPos+1):(iLast-1))
            else
                sSection = ' '
            end if
        else
            ! String begins with a non-blank
            lIsSection = sString(1:1) == '[' .and. sString(iLast:iLast) == ']'
            if(lIsSection) then
                sSection = sString(1:(iLast-1))
            else
                sSection = ' '
            end if
        end if

    end function isSection


    subroutine removeChar(sString, cChar)

        ! Routine arguments
        character(len=*), intent(inout) :: sString
        character, intent(in)           :: cChar

        ! Locals
        integer :: i, j, n

        ! Copy all desired characters, and them only, to the string, in place
        n = len_trim(sString)
        j = 0
        do i = 1, n
            if(sString(i:i) /= cChar) then
                j = j + 1
                if(j /= i) then
                    sString(j:j) = sString(i:i)
                end if
            end if
        end do
        if(j < n) then
            sString((j+1):n) = ' '
        end if

    end subroutine removeChar

end module Config
