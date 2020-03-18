! The m_grid module handles most of the rules surrounding updating
! cells.
! Most routines that another method would need to call are
! meant to be called as type-bound procedures.
module m_grid
implicit none
    
    public
    private::update_grid, alive, alive_threshhold, randomize_grid

    ! Controls whether a cell is 'alive' when randomizing
    ! the grid
    real(kind=4), parameter::alive_threshhold = 0.4

    ! The lifegrid type will contain everything we need
    ! to know about our run's grid.
    type lifegrid
        logical, dimension(:,:), allocatable::grid
        integer::rows
        integer::columns
        
        contains
        
        procedure::update => update_grid
        procedure::randomize => randomize_grid
        
    end type lifegrid
    
    contains

    ! Creates a new grid
    function allocate_grid(rows, columns) result(grid)
    implicit none
        
        type(lifegrid)::grid
        integer, intent(in)::rows, columns
        
            allocate(grid%grid(rows, columns))
            grid%rows = rows
            grid%columns = columns
        
    end function allocate_grid
    
    ! Releases all memory associated with a grid
    subroutine free_grid(self)
    implicit none
    
        type(lifegrid), intent(inout)::self
    
        deallocate(self%grid)
        self%rows = 0
        self%columns = 0
        
    end subroutine free_grid
    
    ! Using the rule set, determines if a grid point
    ! should be alive or dead based on its eight
    ! neighbors
    pure function alive(grid, row, column)
    implicit none
    
        class(lifegrid), intent(in)::grid
        integer, intent(in)::row, column
        logical::alive
        
        integer::rstart, rend, cstart, cend
        integer::neighbor_count
        
        alive = grid%grid(row, column)
        
        ! Calculate the coordinates of the eight (or fewer)
        ! neighbors of our cell of interest
        rstart = max(1, row-1)
        rend = min(grid%rows, row+1)
        cstart = max(1, column-1)
        cend = min(grid%columns, column+1)
        
        ! Since our grid is entirely logicals, we can use the
        ! intrinsic function to count the number of surrounding 
        ! live cells.
        neighbor_count = count(grid%grid(rstart:rend, cstart:cend))
        
        if(alive) then
            ! If it's alive, it was counted above, so decrement the
            ! neighbor_count
            neighbor_count = neighbor_count - 1
            
            ! The cell remains alive with 2 or 3 live neighbors
            alive = (neighbor_count .EQ. 2 .OR. neighbor_count .EQ. 3)
        else
            ! The cell comes to life if there are three live
            ! neighbors
            alive = (neighbor_count .EQ. 3)
        end if
        
    end function alive
    
    ! Cycles through the entire grid and updates the cells row-by-row.
    ! Because any given row will only depend on the old values of the
    ! two surrounding rows, we can update the grid two rows back as we
    ! walk the rows, meaning we don't have to maintain a complete pair
    ! of grids.
    subroutine update_grid(self)
    implicit none
    
        class(lifegrid), intent(inout)::self
        
        ! Need to buffer the previous row's updated values before
        ! we can directly update our grid, hence 'lastline'
        logical, dimension(self%columns)::lastline, thisline
        integer::row, column
        
        lastline = .FALSE.
        
        do row=1, self%rows
            
            do column = 1, self%columns
                thisline(column) = alive(self, row, column)
            end do

            ! After the update of the current row, we can 
            ! update the previous row on the grid since its
            ! original values are no longer needed
            if(row .GT. 1) then
                self%grid(row-1,:) = lastline
            end if
            
            ! Save this line for update next pass
            lastline = thisline
        end do
        
        ! We need to update that last row now.
        self%grid(self%rows,:) = thisline
        
    end subroutine update_grid
    
    ! Randomly populates the grid
    subroutine randomize_grid(self)
    implicit none
    
        class(lifegrid), intent(inout)::self
        
        ! One row of random values
        real(kind=4),dimension(self%columns)::randoms
        integer::row
    
        ! Populate the grid row-by-row rather than allocating
        ! a giant array of real values
        do row = 1, self%rows
            call random_number(randoms)
            self%grid(row,:) = (randoms .LT. alive_threshhold)
        end do
    
    end subroutine randomize_grid
    
end module m_grid
    
