program mmtest
    use mmmod
	implicit none
	real(kind=8), allocatable :: a(:,:), b(:,:), c(:,:)
	integer :: m, n, k
	integer :: t1, t2, rate, i, j
    logical :: bprint

	print *, 'enter matrix size'
	read *, m, k, n

    print *, 'print or not?'
    read *, bprint

	allocate(a(n,k))
	allocate(b(k,m))
	allocate(c(n,m))

	call random_number(a)
	call random_number(b)

    call mkl_set_num_threads(1)
	call system_clock(t1, rate)
    call mm(a,b,c)
	call system_clock(t2)
	print *, 'dgemm: time elapsed = ', real(t2-t1,8)/rate, ' seconds'

    if (bprint) call printc()

	call system_clock(t1, rate)
	c = matmul(a,b)
	call system_clock(t2)
	print *, 'built-in matmul: time elapsed = ', real(t2-t1,8)/rate, ' seconds'
	
    if (bprint) call printc()

	deallocate(a)
	deallocate(b)
	deallocate(c)

contains 
    subroutine printc()
        implicit none
        integer :: j
        print *, 'c = '
        do j = 1, size(c,1)
            print *, c(j, :)
        enddo
    end subroutine
end program mmtest

