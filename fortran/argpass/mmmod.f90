module mmmod

contains
    subroutine mm(A, B, C)
        implicit none
        real(kind=8), intent(in) :: A(:,:), B(:,:)
        real(kind=8), intent(out) :: C(:,:)
        integer :: m, n, k
        m = size(A, 1)
        k = size(A, 2)
        n = size(B, 2)
        call dgemm('N', 'N', m, n, k, 1.d0, A, m, B, k, 0.d0, C, m)
    end subroutine
end module
