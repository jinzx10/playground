module test
    implicit none

    private

    integer :: i1 = 123
    integer :: i2 = 432

    public :: i2, ptt

contains

    subroutine pto
        print *, 'one'
    end subroutine

    subroutine ptt
        print *, 'two'
    end subroutine

end module


program main
    use test

    implicit none

    print *, 'i2 = ', i2
    !print *, 'i1 = ', i1

    call pto

end program
