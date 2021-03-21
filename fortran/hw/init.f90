program main 

    implicit none

    interface
        function init() result (res)
            implicit none
            integer :: res
        end function 
    end interface

    print *, 'init() = ', init()
    print *, 'init() = ', init()
    print *, 'init() = ', init()
    
end program main

function init() result(res)
    implicit none
    integer :: i = 1, res
    i = i + 1
    res = i
end function 
