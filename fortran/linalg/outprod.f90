program outprod
	implicit none
	integer, parameter :: dp = selected_real_kind(15)
	real(kind=dp), allocatable, dimension(:) :: x, y
	real(kind=dp), allocatable, dimension(:, :) :: A
	integer :: nr, nc
	integer :: t1, t2, rate

	print *, 'enter row size'
	read *, nr
	print *, 'enter column size'
	read *, nc

	allocate(x(nr))
	allocate(y(nc))
	allocate(A(nr,nc))

	call random_number(x)
	call random_number(y)

	call system_clock(t1, rate)
	A = matmul(reshape(x,(/nr,1/)), reshape(y, (/1,nc/)))
	call system_clock(t2)
	print *, 'built-in matmul: time elapsed = ', real(t2-t1,dp)/rate, ' seconds'
	
	deallocate(x)
	deallocate(y)
	deallocate(A)

end program
