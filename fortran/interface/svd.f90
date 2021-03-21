program main
    implicit none
    real(kind=8), allocatable :: A(:,:), U(:,:), Sigma(:), VT(:,:)

    interface svd
        subroutine svd_s(A, Sigma)
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), allocatable :: Sigma(:)
        end subroutine svd_s

        subroutine svd_all(A, U, Sigma, VT)
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), allocatable :: U(:,:), Sigma(:), VT(:,:)
        end subroutine svd_all

    end interface svd

    interface svd_econ
        subroutine svd_econ_s(A, Sigma)
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), allocatable :: Sigma(:)
        end subroutine svd_left

        subroutine svd_econ_all(A, U, Sigma, VT)
            real(kind=8), intent(in) :: A(:,:)
            real(kind=8), allocatable :: U(:,:), Sigma(:), VT(:,:)
        end subroutine svd_all

    end interface svd

end program main


subroutine svd_left_right(A, U, Sigma, VT)
    implicit none
    real(kind=8), intent(in) :: A(:,:)
    real(kind=8), allocatable :: U(:,:), Sigma(:), VT(:,:)

end subroutine svd_all

