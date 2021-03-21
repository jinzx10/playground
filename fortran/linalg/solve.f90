program solve
    use svmod

    implicit none
	real(kind=8), allocatable :: A(:,:), b(:), x(:)
    integer :: sz, i

	print *, 'enter matrix size'
    read *, sz

    allocate(A(sz,sz))
    allocate(b(sz))
    allocate(x(sz))

	call random_number(A)
	call random_number(b)
    A = A + transpose(A)

    do i = 1, sz
        A(i,i) = A(i,i) + sz
    enddo

    print *, 'input: '
    call printa()
    print *, 'b = ', b

    call dsolve(x, A, b)

    print *, 'output: '
    call printa()
    print *, 'b = ', b
    print *, 'x = ', x

    call tqsolve(x, A, b)
    print *, 'output: '
    call printa()
    print *, 'b = ', b
    print *, 'x = ', x
contains
    subroutine printa()
        implicit none
        integer :: ia
        print *, 'A = '
        do ia = 1, size(A, 1)
            print *, A(ia, :)
        enddo
    end subroutine
end program
