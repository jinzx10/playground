program array
    implicit none
    integer :: m, n
    integer :: A(3,4)

    call wrapper()

    !real, dimension(3) :: x = (/ 1.0, 2.0, 3.0 /)
    !real, dimension(3) :: y, z, w, u
    !real, dimension(3,3) :: m = 0
    !real :: p(3)

    !real :: a = 0.5, v

    !y = a + x
    !z = x + y
    !w = x * a
    !u = x / a

    !v = sum(x*x)

    !m(1,1) = 1.0
    !m(2,2) = 2.0
    !m(3,3) = 3.0

    !p = m(1,:)
    !p = matmul(p, m)
    !
    !print *, y
    !print *, z
    !print *, w
    !print *, u
    !print *, v
    !print *, m
    !print *, p

end program array

subroutine wrapper()
    implicit none
    integer :: m, n
    integer, allocatable :: A(:,:), v(:)

    interface f
        procedure func
        procedure funcc
    end interface f

    m=3
    n=4
    allocate(A(m,n))
    A=0

    call f(m,n,A,v)
    A=2
    call f(m,n,A)

contains
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

end subroutine wrapper
