program main
    use lsmod
    implicit none
    integer :: flag

    call linesearch(flag)

    print *, 'exit flag = ', flag
    print *, 'alpha = ', alpha

end program 
