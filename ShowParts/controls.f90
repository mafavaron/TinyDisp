! The m_control module is used for controlling the game.  It includes
! both variables for flow control and on-screen elements like menus
! and buttons.
!
! The flow control variables (paused, playing) are all marked 'volatile' 
! because they will regularly be updated by one thread (the user 
! interface thread) and read by the game thread.  The 'volatile' 
! attribute ensures that the actual memory rather than a cached value is
! queried.
module m_control
implicit none

    ! True to pause the game, false to run the game
    logical, volatile::paused
    
    ! True if the game should continue running, false to quit the
    ! program
    logical, volatile::playing
    
    ! The identifiers of two buttons
    integer::start_button, stop_button
    
    ! The identifiers of all our menus
    integer::root_menu, file_menu, game_menu, about_menu
    integer::pause_menuitem, resume_menuitem, reset_menuitem
    
    contains

    ! Initializes flow control for our game
    subroutine init_controls()
    implicit none
    
        paused = .TRUE.
        playing = .TRUE.

    end subroutine init_controls
    
    ! Adds a few menus to our game
    subroutine init_menu()
    use appgraphics
    implicit none
    
    ! For many of these menu items, we don't really need to
    ! know their identifiers...   In fact, we're only storing
    ! some such that we can enable/disable them in the proper
    ! circumstances.
    integer::item_temp
    
        root_menu = addmenu("", MENU_FOR_WINDOW)
        file_menu = addmenu("File", root_menu)
        game_menu = addmenu("Game", root_menu)
        about_menu = addmenu("Help", root_menu)
        
        item_temp = addmenuitem("Save Screenshot...", file_menu, savescreen)
        item_temp = addmenuitem("Quit", file_menu, quitgame)
        
        pause_menuitem = addmenuitem("Pause", game_menu, stopgame)
        resume_menuitem = addmenuitem("Resume", game_menu, startgame)
        reset_menuitem = addmenuitem("Reset Game", game_menu, resetgame)
        
        item_temp = addmenuitem("About...", about_menu, aboutgame)
        
    end subroutine init_menu

    ! Pauses the game
    subroutine stopgame()
    use appgraphics, only: enablebutton, enablemenuitem
    implicit none
    
        paused = .TRUE.
    
        call enablebutton(stop_button, .FALSE.)
        call enablebutton(start_button, .TRUE.)
        
        call enablemenuitem(pause_menuitem, .FALSE.)
        call enablemenuitem(resume_menuitem, .TRUE.)
        call enablemenuitem(reset_menuitem, .TRUE.)
        
    end subroutine stopgame

    ! Starts/resumes the game
    subroutine startgame()
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
        
    end subroutine startgame
    
    ! Quits the program cleanly
    subroutine quitgame()
    implicit none
        
        playing = .FALSE.
        
        ! Need to break free from the idle loop in order
        ! to acknowledge ending the game
        call startgame()
        
    end subroutine quitgame
    
    ! Captures the screen as a bitmap
    subroutine savescreen()
    use appgraphics, only: writeimagefile
    implicit none
    
        ! This call will open a file dialog since we haven't
        ! specified a filename
        call writeimagefile()
    
    end subroutine savescreen
    
    ! Re-fills the grid with random data
    subroutine resetgame()
    use m_globals, only: grid
    use appgraphics, only: stopidle
    implicit none
        
        call grid%randomize()
        
        ! Forces a redraw, but the paused flag is still
        ! set to True, meaning the game won't actually
        ! start
        call stopidle()
        
    end subroutine resetgame

    ! Displays a message box
    subroutine aboutgame
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
        msg = "Conway's Game of Life"//ff//"A demo of AppGraphics by Approximatrix"//ff//ff//&
              "Please feel free to modify and use this demonstration"//ff//&
              "in any way you wish."
        
        call dlgmessage(DIALOG_INFO, msg)
        
    end subroutine aboutgame

end module m_control
