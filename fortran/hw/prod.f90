program test
    implicit none
    integer :: x(5) = (/1,2,3,4,5/)

    print *, product(x(1:5))
    print *, product(x(2:2))
    print *, product(x(4:2))
end program
