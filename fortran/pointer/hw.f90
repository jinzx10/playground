module hw

contains

    subroutine print_arr(a)
        implicit none
        real :: a(:)
        print *, a
    end subroutine print_arr

end module hw

program test
    use hw
    implicit none

    real, target :: a(5)
    real, pointer :: pa(:)

    a = (/1., 2., 3., 4., 5./)

    allocate(pa(5))
    pa = (/1., 2., 3., 4., 5./)
    !pa => a

    call print_arr(pa)

    deallocate(pa)
    nullify(pa)
    nullify(pa)

end program test

