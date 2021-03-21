module linalg
    implicit none

    interface svd_econ
        module procedure dgesvd_econ_USV

    end interface svd_econ

    interface svd
        module procedure dgesvd_USV
        module procedure dgesvd_US
    end interface svd

contains

    subroutine print_mat(M, varname)
        real(kind=8) :: M(:,:)
        character(len=*) :: varname
        integer :: i
        print *, varname, ' = '
        do i = 1, size(M,1)
            print *, M(i,:)
        enddo
    end subroutine
    
    subroutine dgesvd_prototype(A, U, S, VT, jobu, jobvt)
        real(kind=8), intent(in) :: A(:,:)
        character, intent(in) :: jobu, jobvt

        real(kind=8) :: U(:,:), VT(:,:), S(:)

        real(kind=8), allocatable :: work(:), tmp(:,:)
        integer :: m, n, info, lwork, ldvt

        m = size(A, 1)
        n = size(A, 2)

        allocate( work(1) )
        allocate( tmp(m, n) )

        ldvt = n
        if ( jobvt .eq. 'S' ) ldvt = min(m,n)

        tmp = A
        call dgesvd(jobu, jobvt, m, n, tmp, m, S, U, m, VT, ldvt, work, -1, info)

        lwork = work(1)
        deallocate( work )
        allocate( work(lwork) )

        call dgesvd(jobu, jobvt, m, n, tmp, m, S, U, m, VT, ldvt, work, lwork, info)
        if (info .ne. 0) print *, 'dgesvd failed!' 

        deallocate( work )
        deallocate( tmp )

    end subroutine dgesvd_prototype

    subroutine dgesvd_econ_USV(A, U, S, VT)
        real(kind=8), intent(in) :: A(:,:)
        real(kind=8) :: U(:,:), VT(:,:), S(:)
        call dgesvd_prototype(A, U, S, VT, 'S', 'S')
    end subroutine 

    subroutine dgesvd_USV(A, U, S, VT)
        real(kind=8), intent(in) :: A(:,:)
        real(kind=8) :: U(:,:), VT(:,:), S(:)
        call dgesvd_prototype(A, U, S, VT, 'A', 'A')
    end subroutine 

    subroutine dgesvd_US(A, U, S)
        real(kind=8), intent(in) :: A(:,:)
        real(kind=8) :: U(:,:), VT(0,0), S(:)
        call dgesvd_prototype(A, U, S, VT, 'A', 'N')
    end subroutine 



end module linalg
