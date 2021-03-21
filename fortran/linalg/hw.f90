program main
    use linalg

    implicit none

    real(kind=8) :: A(3,5), B(5,3), S(3)
    real(kind=8), allocatable :: U(:,:), VT(:,:)
    integer :: i, j

    A(1,:) = (/ 1.d0, 2.d0, 3.d0, 4.d0, 5.d0 /)
    A(2,:) = (/ 2.d0, 2.d0, 1.d0, 3.d0, 4.d0 /)
    A(3,:) = (/ 2.d0, 5.d0, 3.d0, 1.d0, 8.d0 /)

    B = transpose(A)

    !--------------------------------------------------

    allocate( U(3,3) )
    allocate( VT(5,5) )

    call svd(A, U, S, VT)

    call print_mat(A, 'A')
    print *, 'S = ', S
    call print_mat(U, 'U')
    call print_mat(VT, 'VT')

    deallocate(U)
    deallocate(VT)

    !--------------------------------------------------

    allocate( U(5,5) )
    allocate( VT(3,3) )

    call svd(B, U, S, VT)

    call print_mat(B, 'B')
    print *, 'S = ', S
    call print_mat(U, 'U')
    call print_mat(VT, 'VT')

    deallocate(U)
    deallocate(VT)

    !--------------------------------------------------

    allocate( U(5,3) )
    allocate( VT(3,3) )

    call svd_econ(B, U, S, VT)

    call print_mat(B, 'B')
    print *, 'S = ', S
    call print_mat(U, 'U')
    call print_mat(VT, 'VT')

    deallocate(U)
    deallocate(VT)

    !--------------------------------------------------

    allocate( U(5,3) )
    allocate( VT(3,3) )

    call svd(B, U, S)

    call print_mat(B, 'B')
    print *, 'S = ', S
    call print_mat(U, 'U')
    call print_mat(VT, 'VT')

    deallocate(U)
    deallocate(VT)

end program main
