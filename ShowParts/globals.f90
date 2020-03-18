! The m_globals module is used to store our only globally-accesible
! variable, the grid of cells, rather than passing it around
! everywhere.
module m_globals
use m_grid
implicit none

    type(lifegrid)::grid
    
end module m_globals
