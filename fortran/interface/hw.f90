program main
    implicit none

    call test()

end program main


subroutine test
    implicit none

    interface addup 
        function f2(x1,x2) result(y)
            integer, intent(in) :: x1,x2
            integer :: y
        end function f2

        function f3(x1,x2,x3) result(y)
            integer, intent(in) :: x1,x2,x3
            integer :: y
        end function f3

    end interface addup

    print *, 'test starts'

    write(*,*) addup(2,4,5), addup(3,4)

end subroutine test


function f2(x1,x2) result(y)
    implicit none
    integer, intent(in) :: x1,x2
    integer :: y
    y = x1+x2
end function f2

function f3(x1,x2,x3) result(y)
    implicit none
    integer, intent(in) :: x1,x2,x3
    integer :: y
    y = x1+x2+x3
end function f3
