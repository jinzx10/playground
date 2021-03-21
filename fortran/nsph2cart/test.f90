module scmod
    implicit none
    integer, parameter :: DP = selected_real_kind(8,200)

contains

!=======================================================================
! this subroutine converts an n-spherical coordinate
! on a unit n-sphere to the Cartisian coordinate
!
!  array   size
!----------------
!  nsph    n-1
!  cart     n
!
! the transformation can be expressed as
!
! cart = cumprod([1; sin(nsph)]) .* [cos(nsph); 1] 
!
! or
!
! cart(1)   = cos(nsph(1))
! cart(2)   = sin(nsph(1)) * cos(nsph(2))
! ...
! cart(n-1) = sin(nsph(1)) * ... * sin(nsph(n-2)) * cos(nsph(n-1))
! cart(n)   = sin(nsph(1)) * ... * sin(nsph(n-2)) * sin(nsph(n-1))
!
subroutine nsph2cart(nsph, cart, jacobian)
    implicit none
    !
    real(DP), intent(in)            :: nsph(:)
    real(DP), intent(out)           :: cart(:)
    real(DP), intent(out), optional :: jacobian(:,:)
    !
    integer                         :: n, i, j
    real(DP)                        :: cpsin
    !
    n = size(nsph) + 1
    if (present(jacobian)) jacobian = 0.d0
    !
    cpsin = 1.d0
    do i = 1, n-1
        cart(i) = cpsin * cos(nsph(i))
        cpsin = cpsin * sin(nsph(i))
        if (present(jacobian)) jacobian(i,i) = -cpsin
    enddo
    cart(n) = cpsin 
    !
    if (present(jacobian)) then
        do i = 1, n-1
            cpsin = 1.d0
            do j = i+1, n-1
                jacobian(i,j) = cart(i) * cpsin * cos(nsph(j))
                cpsin = cpsin * sin(nsph(j))
            enddo
            jacobian(i,n) = cart(i) * cpsin
        enddo
    endif
end subroutine nsph2cart
!
!=======================================================================
! this subroutine extends nsph2cart to the complex case
!
! let znsph = [theta; phi] 
!     zcart = [x; y]
!
! and their corresponding sizes follow
!
!  array    size
!-------------------
!  theta    n-1
!   phi     n-1
!  znsph   2*n-2
!-------------------
!    x       n
!    y      n-1
!  zcart   2*n-1
!
! the transformation znsph -> zcart follows a two-step procedure:
!
! 1. compute theta -> cart according to nsph2cart
! 2. x(i) = cart(i) * cos(phi(i))    i = 1 to n-1
!    x(n) = cart(n)
!    y(i) = cart(i) * sin(phi(i))    i = 1 to n-1
!
subroutine znsph2zcart(znsph, zcart, zjacobian)
    implicit none
    !
    real(DP), intent(in)            :: znsph(:)
    real(DP), intent(out)           :: zcart(:)
    real(DP), intent(out), optional :: zjacobian(:,:)
    !
    real(DP), allocatable           :: cart(:), jacobian(:,:)
    integer                         :: n, i
    !
    n = size(znsph)/2 + 1
    !
    allocate( cart(n) )
    !
    if ( present(zjacobian) ) then
        allocate( jacobian(n-1,n) )
        call nsph2cart(znsph(1:n-1), cart, jacobian)
    else
        call nsph2cart(znsph(1:n-1), cart)
    endif
    !
    zcart(1:n-1) = cart(1:n-1) * cos(znsph(n:2*n-2))
    zcart(n) = cart(n)
    zcart(n+1:2*n-1) = cart(1:n-1) * sin(znsph(n:2*n-2))
    !
    deallocate(cart)
    !
    if ( present(zjacobian) ) then
        zjacobian = 0.d0
        do i = 1, n-1
            zjacobian(n-1+i, i) = -zcart(n+i) ! dx/dphi
            zjacobian(n-1+i, n+i) = zcart(i)  ! dy/dphi
        enddo
        !
        do i = 1, n-1
            zjacobian(1:n-1,i) = jacobian(:,i) * cos(znsph(n-1+i))   ! dx/dtheta
            zjacobian(1:n-1,n+i) = jacobian(:,i) * sin(znsph(n-1+i)) ! dy/dtheta
        enddo
        zjacobian(1:n-1,n) = jacobian(:,n)
        !
        deallocate(jacobian)
    endif
    !
end subroutine znsph2zcart
!
subroutine test_jr()
    implicit none
    integer :: n, i 
    real(DP), allocatable :: nsph(:), cart(:), J(:,:)
    
    n = 6
    allocate(nsph(n-1))
    allocate(cart(n))
    allocate(J(n-1,n))

    nsph = (/ 0.1d0, 1.d0, 0.5d0, 0.7d0, 0.8d0/)
    print *, 'nsph = ', nsph

    call nsph2cart(nsph, cart, J)

    print *, 'cart = ', cart

    print *, 'J = '
    do i = 1, n-1
        print *, J(i,:)
    enddo

    deallocate(nsph)
    deallocate(J)
end subroutine test_jr

subroutine test_zjr()
    implicit none
    integer :: n, i 
    real(DP), allocatable :: znsph(:), zcart(:), zjacobian(:,:)

    n = 4
    allocate(zcart(2*n-1))
    allocate(znsph(2*n-2))
    allocate(zjacobian(2*n-2, 2*n-1))

    !znsph = (/ 0.5d0, 0.7d0, 0.8d0, 1.d0, 2.d0, 3.d0 /)
    znsph = 0.d0
    call znsph2zcart(znsph, zcart, zjacobian)

    print *, 'znsph = ', znsph
    print *, 'zcart = ', zcart
    print *, 'zjacobian = '
    do i=1,2*n-2
        print *, zjacobian(i,:)
    enddo
    deallocate(zcart)
    deallocate(znsph)
    deallocate(zjacobian)
end subroutine
end module

program test
    use scmod
    implicit none
    real(DP), allocatable :: z(:), nsph(:), J(:,:), nsphtmp(:)
    real(DP) :: hpi
    integer :: n, i
    n = 4
    hpi = acos(-1.d0) / 2.d0

    allocate(z(n), nsph(n-1), J(n-1,n), nsphtmp(n-1))
    nsph = (/1.d0, 2.d0, 3.d0/)

    call nsph2cart( nsph, z, J )

    print *, 'nsph = ', nsph
    print *, 'z = ', z

    print *, 'J = '
    do i = 1, size(J,1)
        print *, J(i,:)
    enddo

    nsphtmp = nsph
    nsphtmp(1) = nsphtmp(1) + hpi
    call nsph2cart(nsphtmp, z)
    print *, 'z = ', z

    nsphtmp = nsph
    nsphtmp(2) = nsphtmp(2) + hpi
    call nsph2cart(nsphtmp, z)
    print *, 'z = ', z

    nsphtmp = nsph
    nsphtmp(3) = nsphtmp(3) + hpi
    call nsph2cart(nsphtmp, z)
    print *, 'z = ', z

end program

