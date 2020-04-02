! Main program of TinyDisp
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program TinyDisp

    use omp_lib
	use Meteo
	
    implicit none
	
	! Locals
    integer:: thread_id, nthreads
	
	! Get input parameters
	
	! Get configuration
	
	! Read meteo data, and expand it to the desired time step
	
	! Initialize particles count
	
	! Main loop: iterate over all time steps, and simulate transport and diffusion

    !$omp parallel private(thread_id)

    thread_id = omp_get_thread_num()
    write (*,*) 'Hello World from thread', thread_id

    !$omp barrier
    if ( thread_id == 0 ) then
        nthreads = omp_get_num_threads()
        write (*,*) 'There are', nthreads, 'threads'
    end if
    
    !$omp end parallel
	
		! Emit new particles
		
		! Move particles
		
		! Save active particles

end program TinyDisp
