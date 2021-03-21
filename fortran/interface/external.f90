program array
    implicit none

    call wrapper()

end program array

subroutine wrapper()
    implicit none
    integer :: m, n
    integer, allocatable :: A(:,:), v(:)

    interface f
        subroutine func(m,n,A,v)
            integer :: m, n
            integer :: A(:,:), v(:)
        end subroutine func

        subroutine funcc(m,n,A)
            integer :: m, n
            integer :: A(:,:)
        end subroutine funcc

    end interface f

    m=3
    n=4
    allocate(A(m,n))
    A=0

    call f(m,n,A,v)
    A=2
    call f(m,n,A)

    print *, 'wrapper: v = ' , v

end subroutine wrapper

subroutine funcc(m,n,A)
    integer :: m, n
    integer :: A(:,:)

    print *, 'A=', A

end subroutine funcc

subroutine func(m,n,A,v)
    integer :: m, n
    integer :: A(:,:)
    integer, allocatable :: v(:)

    if (.not. allocated(v)) allocate(v(min(m,n)))
    if (size(v) .ne. min(m,n)) then
        deallocate(v)
        allocate(v(min(m,n)))
    endif
    v=1
    print *, 'A=', A
    print *, 'v=', v

end subroutine func
