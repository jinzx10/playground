module tqexpm

    integer, parameter :: dp = selected_real_kind(8,200)
    real(dp), parameter :: eps = 1.d-16

contains

subroutine rtqexpm(x, Q)
    !
    ! Given an input vector x, let 
    !
    ! A = [ 0, -x^T;
    !       x,   0  ]
    !
    ! be an anti-symmetric matrix, this subroutine calculates
    !
    ! Q = expm(A)
    !
    ! which makes use of the fact that the eigen-decomposition 
    ! of A can be obtained by a Householder reflection followed 
    ! by a 2x2 diagonalization.
    !
    implicit none
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: Q(:,:)
    !
    integer :: sz, i
    real(dp), external :: dnrm2
    real(dp) :: r
    real(dp), allocatable :: n(:)
    !
    sz = size(x)
    Q = 0.d0
    !
    r = dnrm2(sz, x, 1)
    if (r .lt. eps) then
        do i = 1, sz+1
            Q(i,i) = 1.d0
        enddo
        return
    endif
    if (x(1) .gt. 0) r = -r
    !
    allocate(n(sz))
    call dcopy(sz, x, 1, n, 1)
    n(1) = n(1) - r
    call drscl(sz, sqrt(2.d0*r*(r-x(1))), n, 1)
    !
    Q(1 ,1 ) = cos(r)
    Q(2 ,1 ) = sin(r) * (1.d0-2.d0*n(1)**2)
    Q(3:,1 ) = -2.d0 * sin(r) * n(1) * n(2:)
    Q(1 ,2:) = -Q(2:,1)
    Q(2 ,2 ) = (cos(r)-1.d0) * (1.d0-2.d0*n(1)**2)**2 + 1.d0
    Q(3:,2 ) = n(1) * (1.d0-cos(r)) * (2.d0-4.d0*n(1)**2) * n(2:)
    !
    do i=1,sz-1
        Q(2+i,2+i) = 1.d0
    enddo
    !
    call dsyr('l', sz-1, 4.d0*(cos(r)-1.d0)*n(1)**2, n(2), 1, Q(3,3), sz+1)
    do i=2,sz
        call dcopy(sz+1-i, Q(i+1,i), 1, Q(i,i+1), sz+1)
    enddo
    !
    deallocate(n)
    !
end subroutine


subroutine ctqexpm(z, U)
    implicit none
    complex(dp), intent(in) :: z(:)
    complex(dp), intent(out) :: U(:,:)
    !
    integer :: sz, i
    real(dp), external :: dnrm2
    real(dp) :: r
    complex(dp), allocatable :: n(:)
    complex(dp) :: w, phw
    !
    sz = size(z)
    U = (0.d0,0.d0)
    !
    r = dnrm2(2*sz, z, 1)
    if (r .lt. eps) then
        do i = 1, sz+1
            U(i,i) = (1.d0,0.d0)
        enddo
        return
    endif
    w = -r * exp( (0.d0,1.d0) * atan2(aimag(z(1)),real(z(1))) )
    !
    allocate(n(sz))
    call zcopy(sz, z, 1, n, 1)
    n(1) = n(1) - w
    call zdrscl(sz, sqrt(2.d0*r*(r+abs(z(1)))), n, 1)
    !
    phw = exp( (0.d0,1.d0) * atan2(aimag(w),real(w)) )
    !
    U(1 ,1 ) = cos(r)
    U(2 ,1 ) = phw * sin(r) * (1.d0-2.d0*abs(n(1))**2)
    U(3:,1 ) = -2.d0 * phw * sin(r) * conjg(n(1)) * n(2:)
    U(1 ,2:) = -conjg(U(2:,1))
    U(2 ,2 ) = (cos(r)-1.d0) * (1.d0-2.d0*abs(n(1))**2)**2 + 1.d0
    U(3:,2 ) = conjg(n(1)) * (1.d0-cos(r)) * (2.d0-4.d0*abs(n(1))**2) * n(2:)
    !
    do i=1,sz-1
        U(2+i,2+i) = (1.d0,0.d0)
    enddo
    !
    call zher('l', sz-1, 4.d0*(cos(r)-1.d0)*abs(n(1))**2, n(2), 1, U(3,3), sz+1)
    do i=2,sz
        call zcopy(sz+1-i, U(i+1,i), 1, U(i,i+1), sz+1)
        call zlacgv(sz+1-i, U(i,i+1), sz+1)
    enddo
    !
    deallocate(n)
    !
end subroutine ctqexpm


end module tqexpm

program test
    use tqexpm
    implicit none

    real(dp), allocatable :: x(:), A(:,:)
    complex(dp), allocatable :: y(:), B(:,:), C(:,:)
    integer :: sz,i 

    sz = 5
    allocate(x(sz), A(sz+1,sz+1))

    !call random_number(x)
    x = (/1.d0, 2.d0, 3.d0, 4.d0, 5.d0/)
    print *, 'x = ', x
    print *, ''

    call rtqexpm(x, A)

    do i = 1,sz+1
        print *, A(i,:)
    enddo


    allocate(y(sz), B(sz+1,sz+1))

    !call random_number(x)
    do i=1,sz
        y(i) = i+(0.d0,1.d0)*(i+1)
    enddo
    print *, 'y = ', y
    print *, ''

    call ctqexpm(y, B)

    do i = 1,sz+1
        print *, real(B(i,:))
    enddo
    print *, ''
    do i = 1,sz+1
        print *, aimag(B(i,:))
    enddo
    print *, ''

    allocate(C(sz+1,sz+1))
    C = matmul(B, conjg(transpose(B)))
    do i = 1,sz+1
        print *, real(C(i,:))
    enddo
    print *, ''
    print *, maxval(aimag(C))
end program

