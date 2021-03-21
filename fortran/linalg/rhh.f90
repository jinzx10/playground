program test
    implicit none
    integer, parameter :: dp = selected_real_kind(8,200)
    real(dp), allocatable :: x(:), n(:)
    integer :: sz
    real(dp) :: tmp
    real(dp), external :: dnrm2, ddot

    sz = 5
    allocate(x(sz), n(sz))

    call random_number(x)
    print *, x

    tmp = dnrm2(sz, x, 1)

    n = 0.d0
    n(1) = tmp
    call daxpy(sz, -1.d0, x, 1, n, 1)

    tmp = dnrm2(sz, n, 1)
    call dscal(sz, 1.d0/tmp, n, 1)

    tmp = ddot(sz, n, 1, x, 1)
    call daxpy(sz, -2.d0*tmp, n, 1, x, 1)

    print *, x


end program
