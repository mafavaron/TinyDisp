program life
use m_grid
use m_screen
use m_control
use m_globals
use appgraphics, only: startidle, loop, closewindow, ALL_WINDOWS
implicit none

    integer::screen
    
    ! These calls only initialize our game
    call init_random()
    call init_controls()
    
    ! Start with a 800x600 pixel drawing surface
    screen = init_screen(800, 600)
    
    call init_menu()
    
    ! For this demo, we'll use a fixed grid and
    ! randomly populate cells
    grid = allocate_grid(100,100)
    call grid%randomize()
    
    ! This call actually just pauses our game
    ! before entering the game loop
    call stopgame()
    
    
    do while(playing)
    
        ! This routine will handle actually drawing the
        ! current grid to the screen
        call draw_grid(grid)
        
        ! Check if we're paused.  If paused, sit in an
        ! infinite wait state (using 'loop' from
        ! AppGraphics) until the user starts the game.
        ! If we are playing, idle each update round for
        ! 50 milliseconds.
        if(paused) then
            call loop()
        elseif(.NOT. playing) then
            exit
        else
            call startidle(50)
        end if
        
        ! This type-bound procedure call will update the
        ! grid based on the rules of Conway's Game
        call grid%update()
    end do
    
    ! After ending the game, close all AppGraphics
    ! windows (there should be only one).
    call closewindow(ALL_WINDOWS)
    
    contains
    
    ! Basic routine to initialize the random number
    ! generator based on the current clock time
    subroutine init_random()
    implicit none
    
        integer, dimension(:), allocatable :: seed
        integer :: values(8), k
            
        call date_and_time(values=values)
        call random_seed(size=k)
        
        allocate(seed(k))
        seed = values(8)
        call random_seed(put=seed)
        
    end subroutine init_random
end program life
