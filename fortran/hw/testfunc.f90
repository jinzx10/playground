module testfunc

    implicit none

contains

    function quad1(x) result(res)
        real(kind=8) :: x, res
        res = x*x - 1.0d0*x - 1.0d0
    end function quad1

    function quad2(x) result(res)
        real(kind=8), dimension(:), intent(in) :: x
        real(kind=8), dimension(size(x)) :: res

        res(1) = x(1) - 2.d0*x(2)
        res(2) = x(2)*x(2) + x(1) - x(2) - 1.d0
    end function quad2

    subroutine printarray(x)
        real(kind=8), dimension(:), intent(in) :: x
        real(kind=8), dimension(size(x)) :: x2

        x2 = x*x
        print *, x2
    end subroutine printarray
end module testfunc
