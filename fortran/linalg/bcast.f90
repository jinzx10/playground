program test
    implicit none
    real(kind=8) :: A(5,5), x(5), y(1,5), B(5,5)
    integer :: i

    A = 1.d0
    x = 2.d0
    y = 3.d0

    B = A*y

    do i = 1, 5
        print *, B(i,:)
    enddo

end program
