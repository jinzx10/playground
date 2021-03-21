program main

    implicit none

    write(*,*) diff1(testfunc, 2.d0)

contains
    function testfunc(x) result (res)
        real(kind=8) :: x, res
        res = x*x
    end function

    function diff1(func, x0) result (df)
        real(kind=8) :: x0, fp, fm, dx=1.0d-3, df

        interface
            function func(x)
                real(kind=8) :: x, func
            end function
        end interface

        fp = func(x0+dx)
        fm = func(x0-dx)
        df = (fp-fm)/2.0d0/dx

    end function diff1

end program main
