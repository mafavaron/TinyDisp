! Controls.f90 - This Fortran module contains the whole
!                user-controlled application status
!
module Controls

    use appgraphics
    use iso_c_binding
    use Particles

    implicit none
    
    ! Status variables
    logical, volatile   :: lPaused
    logical, volatile   :: lRunning
    
    ! Button identifiers
    integer     :: iStartButton
    integer     :: iStopButton
    
    ! Menu items
    integer     :: iRootMenu
    integer     :: iFileMenu
    integer     :: iRunMenu
    integer     :: iAboutMenu
    integer     :: iPauseMenuItem
    integer     :: iResumeMenuItem
    integer     :: iResetMenuItem
    
contains

    subroutine init_controls()
    
        implicit none
        
        lPaused  = .true.
        lRunning = .true.

    end subroutine init_controls
    
    
    subroutine init_menu()
    
        ! Routine arguments
        ! -none-
        
        ! Locals
        integer::iTempItem  ! Used for menu items whose identifier can be ignored
        
        ! Menu
        iRootMenu  = addmenu("", MENU_FOR_WINDOW)
        iFileMenu  = addmenu("File", iRootMenu)
        iRunMenu   = addmenu("Run", iRootMenu)
        iAboutMenu = addmenu("Help", iRootMenu)
        
        ! Menu items
        iTempItem       = addmenuitem("Save Screenshot...", iFileMenu,  save_screen)
        iTempItem       = addmenuitem("Quit",               iFileMenu,  quit_run)
        iPauseMenuItem  = addmenuitem("Pause",              iRunMenu,   stop_run)
        iResumeMenuItem = addmenuitem("Resume",             iRunMenu,   start_run)
        iResetMenuitem  = addmenuitem("Reset Game",         iRunMenu,   reset_run)
        iTempItem       = addmenuitem("About...",           iAboutMenu, about_run)
        
    end subroutine init_menu
    
    
     subroutine start_run()
    
        ! Routine arguments
        ! -none-
        
        ! Locals
        ! -none-
        
        ! Force state to "running"
        paused = .false.
        
        ! Release the idle loop so that other things can be made
        call stopidle()

        ! Finalize state forcing to "running"
        call enablebutton(iStopButton,  .true.)
        call enablebutton(iStartButton, .false.)
        call enablemenuitem(iPauseMenuItem,  .true.)
        call enablemenuitem(iResumeMenuItem, .false.)
        call enablemenuitem(iResetMenuItem,  .false.)
        
    end subroutine start_run
    
    
    subroutine stop_run()
    
        ! Routine arguments
        ! -none-
        
        ! Locals
        ! -none-
        
        ! Force state to "paused"
        lPaused = .TRUE.
        call enablebutton(iStopButton,  .false.)
        call enablebutton(iStartButton, .true.)
        call enablemenuitem(iPauseMenuItem,  .false.)
        call enablemenuitem(iResumeMenuItem, .true.)
        call enablemenuitem(iResetMenuItem,  .true.)
        
    end subroutine stop_run


    subroutine quit_run()
        
        ! Routine arguments
        ! -none-
        
        ! Locals
        ! -none-
        
        ! Force state to "not running", to allow
        ! orderly breaking out of the idle loop
        lRunning = .false.
        
        ! By doing so, the loop will break
        ! but as lRunning==.false. an orderly
        ! shutdown will occur
        call start_run()
        
    end subroutine quit_run


    subroutine save_screen()
    
        ! Open the screen save dialog
        call writeimagefile()
    
    end subroutine save_screen
    
    
    subroutine reset_run(tPart)
    
        ! Routine arguments
        type(ParticlePoolType), intent(inout)   :: tPart
        
        ! Initialize run
        call tPart % start()
        
        ! Forces a redraw, but the paused flag is still
        ! set to True, meaning the run won't actually
        ! start
        call stopidle()
        
    end subroutine reset_run


    subroutine about_run

        character(2)    :: ff
        character(512)  :: msg
        
        ! To create multiple lines in our message box, we need to have
        ! a \r\n for windows.  For brevity, we can create that variable
        ! here to hold the two characters.
        ff = C_CARRIAGE_RETURN//C_NEW_LINE
        
        msg = repeat(' ', 512)
        msg = "Conway's Game of Life"//ff//"A demo of AppGraphics by Approximatrix"//ff//ff//&
              "Please feel free to modify and use this demonstration"//ff//&
              "in any way you wish."
        
        call dlgmessage(DIALOG_INFO, msg)
        
    end subroutine about_run
    
end module Controls
