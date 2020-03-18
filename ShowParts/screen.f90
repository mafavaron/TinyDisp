! The m_screen module handles all drawing and updates to the screen.
module m_screen
implicit none

    ! A status bar height
    integer(kind=4), parameter::status_height = 40

    contains
    
    ! Initializes the screen by creating our window
    function init_screen(width, height)
    use m_control
    use appgraphics
    implicit none
    
        integer::init_screen
        integer, intent(in)::width, height
    
        ! Note that we've enabled double-buffering of the screen for performance
        init_screen = initwindow(width, height, title="Conway's Game of Life", dbflag=.TRUE., closeflag=.TRUE.)
        
        ! Configure some colors and drawing parameters
        call setcolor(WHITE)
        call setfillstyle(SOLID_FILL, WHITE)
        call setbkcolor(BLACK)
        
        call setmatchthemetextstyle()
        
        ! Create two buttons on the screen to start and stop the game
        start_button = createbutton(10, getmaxy( )- status_height*9/10, &
                                    75, status_height*4/5, &
                                    "Start", startgame)
                                    
        stop_button = createbutton(85, getmaxy() - status_height*9/10, &
                                   75, status_height*4/5, &
                                   "Stop", stopgame)
        
        call settextstyle(SERIF_FONT, HORIZ_DIR, status_height*9/10)
        
    end function init_screen
    
    ! Actually draws the grid, first to memory, then flips the buffers
    ! to display it.  This routine is drastically faster than drawing
    ! it directly because direct drawing forces a screen refresh after
    ! every operation.
    subroutine draw_grid(grid)
    use m_grid
    use appgraphics
    implicit none
    
        class(lifegrid), intent(in)::grid
        real(kind=4)::steprow,stepcolumn
        integer::row,column
        real(kind=4)::screenrow, screencolumn
        character(80)::statusline
        
        ! Calculate, in floating point terms, the width and height
        ! of every cell based on grid size and screen size.  This way we
        ! can change either parameter without having to change the
        ! drawing routine directly.
        steprow = real(getmaxy()-status_height-1, 4) / real(grid%rows, 4)
        stepcolumn = real(getmaxx(), 4) / real(grid%columns, 4)
        
        screenrow = 0

        ! Fills the window with the background color
        call clearviewport()
        
        ! Walk through the grid and draw any live cells
        do row = 1,grid%rows
            screencolumn = 0
            do column = 1,grid%columns
                if(grid%grid(row,column)) then
                    call bar(floor(screencolumn), floor(screenrow), &
                             ceiling(screencolumn+stepcolumn), &
                             ceiling(screenrow+steprow))
                end if
                
                screencolumn = screencolumn + stepcolumn
            end do
            screenrow = screenrow + steprow
        end do

        ! Line above the status area
        call line(0, getmaxy() - status_height, getmaxx(), getmaxy() - status_height)

        ! A count of live cells
        statusline = repeat(' ', 80)
        Write(statusline, "('Live Cell Count: ', I8)") count(grid%grid)
        call settextxy(getmaxx() - textwidth(statusline) - 20,  getmaxy()-status_height*19/20)
        call outtext(statusline)
        
        ! After our drawing is complete, swap buffers to display it
        call swapbuffers()
        
    end subroutine draw_grid

end module m_screen
