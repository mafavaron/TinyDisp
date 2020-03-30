! Test driver for module "NormalDeviates.f90".
!
! Copyright 2020 by Servizi Territorio srl
!                   This is open-source software, covered by the MIT license.
!
! Author: Patti M. Favaron
!
program Test_NormalDeviates

	use NormalDeviates
	
	implicit none
	
	! Locals
	integer					:: iRetCode
	integer					:: i
	real, dimension(100000)	:: rvNormU
	real, dimension(100000)	:: rvNormV
	real, dimension(100000)	:: rvNormW
	real, parameter			:: rU  =  0.
	real, parameter			:: rV  =  1.
	real, parameter			:: rW  =  4.
	real, parameter			:: rUU =  0.25
	real, parameter			:: rVV =  0.25
	real, parameter			:: rWW =  0.25
	real, parameter			:: rUV = -0.1
	real, parameter			:: rUW =  0.1
	real, parameter			:: rVW =  0.05
	
	! Generate a univariate sample N(0,1)
	iRetCode = Norm(rvNormU)
	open(10, file='Norm_Uni.csv', status='unknown', action='write')
	do i = 1, size(rvNormU)
		write(10, "(e15.7)") rvNormU(i)
	end do
	close(10)

	! Generate a multivariate sample
	iRetCode = MultiNorm(rU, rV, rW, rUU, rVV, rWW, rUV, rUW, rVW, rvNormU, rvNormV, rvNormW)
	open(10, file='Norm_Mul.csv', status='unknown', action='write')
	do i = 1, size(rvNormU)
		write(10, "(e15.7,2(',',e15.7))") rvNormU(i), rvNormV(i), rvNormW(i)
	end do
	close(10)

end program Test_NormalDeviates
