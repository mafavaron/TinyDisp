! Controls.f90 - This Fortran module contains the whole
!                user-controlled application status
!
module Controls

    use appgraphics

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

end module Controls
