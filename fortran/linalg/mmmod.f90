module mmmod
    implicit none

    interface mm
        module procedure dmm
        !module procedure zmm
    end interface

contains
    subroutine dmm(A, B, C)
        implicit none
        real(kind=8) :: A(:,:), B(:,:), C(:,:)
        integer :: m, k, n
        m = size(A,1)
        k = size(A,2)
        if (k .ne. size(B,1)) then
            print *, 'incompatible sizes'
            return
        endif
        n = size(B,2)
	    call dgemm('n', 'n', m, n, k, 1.d0, A, m, B, k, 0.d0, C, m)
    end subroutine
end module
