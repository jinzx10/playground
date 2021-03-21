module svmod
    implicit none

contains

    subroutine dsolve(x, A, b)
        implicit none
        real(kind=8), intent(in) :: A(:,:), b(:)
        real(kind=8) :: x(:)
        real(kind=8), allocatable :: work(:), Acopy(:,:)
        integer :: sz, lwork, info
        integer, allocatable :: ipiv(:)
        sz = size(A,1)
        if ( sz .ne. size(A,2) .or. sz .ne. size(b) ) then
            print *, 'invalid size'
            return
        endif
        !
        allocate(Acopy(sz,sz))
        Acopy = A
        x = b
        allocate(ipiv(sz))
        !
        ! work space query
        allocate(work(1))
        lwork = -1
        call dsysv('U', sz, 1, Acopy, sz, ipiv, x, sz, work, lwork, info)
        !
        ! work space allocation
        lwork = work(1)
        deallocate(work)
        allocate(work(lwork))
        !
        ! solve
        call dsysv('U', sz, 1, Acopy, sz, ipiv, x, sz, work, lwork, info)
        if (info .ne. 0) print *, 'dsysv failed.'
        !
        deallocate(Acopy)
        deallocate(work)
        deallocate(ipiv)
    end subroutine

    subroutine tqsolve(x, A, b)
        implicit none
        real(kind=8), intent(in) :: A(:,:), b(:)
        real(kind=8) :: x(:)
        real(kind=8), allocatable :: Acopy(:,:)
        integer :: sz, info
        sz = size(A,1)
        if ( sz .ne. size(A,2) .or. sz .ne. size(b) ) then
            print *, 'invalid size'
            return
        endif
        !
        allocate(Acopy(sz,sz))
        Acopy = A
        x = b
        !
        ! solve
        call dposv('U', sz, 1, Acopy, sz, x, sz, info)
        if (info .ne. 0) print *, 'dposv failed.'
        !
        deallocate(Acopy)
    end subroutine tqsolve
end module
