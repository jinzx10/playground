program test_matmul
	implicit none
	integer, parameter :: dp = selected_real_kind(15)
	real(kind=dp), allocatable, dimension(:, :) :: a, b, c
	integer :: sz
	integer :: t1, t2, rate

	print *, 'enter matrix size'
	read *, sz

	allocate(a(sz,sz), b(sz,sz), c(sz,sz))

	call random_number(a)
	call random_number(b)

    !call mkl_set_num_threads(4)
	call system_clock(t1, rate)
	call dgemm('n', 'n', sz, sz, sz, 1.d0, a, sz, b, sz, 0.d0, c, sz)
	call system_clock(t2)
	print *, 'dgemm: time elapsed = ', real(t2-t1,dp)/rate, ' seconds'

	call system_clock(t1, rate)
	c = matmul(a,b)
	call system_clock(t2)
	print *, 'built-in matmul: time elapsed = ', real(t2-t1,dp)/rate, ' seconds'
	
	deallocate(a, b, c)

end program test_matmul

