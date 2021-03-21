program svdtest
    use svd
    implicit none
    real(kind=8), allocatable :: A(:,:), U(:,:), VT(:,:), S(:), D(:)
    complex(DP), allocatable :: Z(:,:), X(:,:), YT(:,:)
    integer :: m, n, i, r, mode
    logical :: is_econ
    character :: lr

    print *, 'm, n:'
    read *, m, n
    print *, 'is econ?'
    read *, is_econ
    print *, 'how many singular vectors?'
    read *, mode

    if (mode .eq. 1) then
        print *, 'left or right?'
        read *, lr
    endif

    allocate(A(m,n))
    call random_number(A)

    allocate(Z(m,n))
    Z = cmplx(1.d0, 2.d0)

    if (is_econ) then
        r = min(m,n)
        allocate( U(m,r), VT(r,n), S(r) )
        allocate( X(m,r), YT(r,n), D(r) )
    else
        allocate( U(m,m), VT(n,n), S(min(m,n)) )
        allocate( X(m,m), YT(n,n), D(min(m,n)) )
    endif

    select case(mode)
    case(0)
        call tqsvd(A, S)
        call tqsvd(Z, D)
    case(1)
        if (lr .eq. 'L') then
            call tqsvd(A, S, U, lr, is_econ)
            call tqsvd(Z, D, X, lr, is_econ)
        else
            call tqsvd(A, S, VT, lr, is_econ)
            call tqsvd(Z, D, YT, lr, is_econ)
        endif
    case(2)
        call tqsvd(A, U, S, VT, is_econ)
        call tqsvd(Z, X, D, YT, is_econ)
    end select

    print *, 'A = '
    do i=1,m
        print *, A(i,:)
    enddo
    print *, ''

    print *, 'U = '
    do i=1,m
        print *, U(i,:)
    enddo
    print *, ''

    print *, 'VT = '
    do i=1,size(VT,1)
        print *, VT(i,:)
    enddo
    print *, ''

    print *, 'S = ', S

    print *, 'Z = '
    do i=1,m
        print *, Z(i,:)
    enddo
    print *, ''

    print *, 'X = '
    do i=1,m
        print *, X(i,:)
    enddo
    print *, ''

    print *, 'YT = '
    do i=1,size(YT,1)
        print *, YT(i,:)
    enddo
    print *, ''

    print *, 'D = ', D

end program
