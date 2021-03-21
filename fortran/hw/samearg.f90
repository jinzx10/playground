program samearg
    use samemod
    implicit none
    real(kind=8) :: a, b, c, z(2,2), x(2,2)
    integer :: i

    call random_number(x)
    call random_number(z)

    print *, 'x = '
    do i = 1,2
        print *, x(i,:)
    enddo

    print *, 'z = '
    do i = 1,2
        print *, z(i,:)
    enddo

    call test(z, z)

    print *, 'z = '
    do i = 1,2
        print *, z(i,:)
    enddo


end program


