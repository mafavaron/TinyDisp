! The m_control module is used for controlling the run.  It includes
! both variables for flow control and on-screen elements like menus
! and buttons.
!
! The flow control variables (paused, playing) are all marked 'volatile' 
! because they will regularly be updated by one thread (the user 
! interface thread) and read by the run thread.  The 'volatile' 
! attribute ensures that the actual memory rather than a cached value is
! queried.
module m_control
implicit none

    ! True to pause the run, false to run the run
    logical, volatile::paused
    
    ! True if the run should continue running, false to quit the
    ! program
    logical, volatile::playing
    
    ! The identifiers of two buttons
    integer::start_button, stop_button
    
    ! The identifiers of all our menus
    integer::root_menu, file_menu, run_menu, about_menu
    integer::pause_menuitem, resume_menuitem, reset_menuitem
    
    contains

    ! Initializes flow control for our run
    subroutine init_controls()
    implicit none
    
        paused = .TRUE.
        playing = .TRUE.

    end subroutine init_controls
    
    ! Adds a few menus to our run
    subroutine init_menu()
    use appgraphics
    implicit none
    
    ! For many of these menu items, we don't really need to
    ! know their identifiers...   In fact, we're only storing
    ! some such that we can enable/disable them in the proper
    ! circumstances.
    integer::item_temp
    
        root_menu  = addmenu("", MENU_FOR_WINDOW)
        file_menu  = addmenu("File", root_menu)
        run_menu   = addmenu("Run",  root_menu)
        about_menu = addmenu("Help", root_menu)
        
        item_temp = addmenuitem("Save Screenshot...", file_menu, savescreen)
        item_temp = addmenuitem("Quit", file_menu, quitrun)
        
        pause_menuitem  = addmenuitem("Pause", run_menu, stoprun)
        resume_menuitem = addmenuitem("Resume", run_menu, startrun)
        reset_menuitem  = addmenuitem("Reset Run", run_menu, resetrun)
        
        item_temp = addmenuitem("About...", about_menu, aboutrun)
        
    end subroutine init_menu


    subroutine stoprun()
    use appgraphics, only: enablebutton, enablemenuitem
    implicit none
    
        paused = .TRUE.
    
        call enablebutton(stop_button, .FALSE.)
        call enablebutton(start_button, .TRUE.)
        
        call enablemenuitem(pause_menuitem, .FALSE.)
        call enablemenuitem(resume_menuitem, .TRUE.)
        call enablemenuitem(reset_menuitem, .TRUE.)
        
    end subroutine stoprun


    subroutine startrun()
    use appgraphics, only: stopidle, enablebutton, enablemenuitem
    implicit none
    
        paused = .FALSE.
        
        ! Releases the idle 'loop' call in our main program
        call stopidle()
        
        call enablebutton(stop_button, .TRUE.)
        call enablebutton(start_button, .FALSE.)
        
        call enablemenuitem(pause_menuitem, .TRUE.)
        call enablemenuitem(resume_menuitem, .FALSE.)
        call enablemenuitem(reset_menuitem, .FALSE.)
        
    end subroutine startrun
    
    ! Quits the program cleanly
    subroutine quitrun()
    implicit none
        
        playing = .FALSE.
        
        ! Need to break free from the idle loop in order
        ! to acknowledge ending the run
        call startrun()
        
    end subroutine quitrun
    
    ! Captures the screen as a bitmap
    subroutine savescreen()
    use appgraphics, only: writeimagefile
    implicit none
    
        ! This call will open a file dialog since we haven't
        ! specified a filename
        call writeimagefile()
    
    end subroutine savescreen
    
    ! Re-fills the grid with random data
    subroutine resetrun()
    use m_globals, only: grid
    use appgraphics, only: stopidle
    implicit none
        
        call grid%randomize()
        
        ! Forces a redraw, but the paused flag is still
        ! set to True, meaning the run won't actually
        ! start
        call stoprun()
        
    end subroutine resetrun

    ! Displays a message box
    subroutine aboutrun
    use appgraphics
    use iso_c_binding
    implicit none
        
        character(2)::ff
        character(512)::msg
        
        ! To create multiple lines in our message box, we need to have
        ! a \r\n for windows.  For brevity, we can create that variable
        ! here to hold the two characters.
        ff = C_CARRIAGE_RETURN//C_NEW_LINE
        
        msg = repeat(' ', 512)
        msg = "Particle visualizer"//ff//"A demo of AppGraphics by Approximatrix"//ff//ff//&
              "Please feel free to modify and use this demonstration"//ff//&
              "in any way you wish."
        
        call dlgmessage(DIALOG_INFO, msg)
        
    end subroutine aboutrun

end module m_control
