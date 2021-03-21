module svd

    integer, PARAMETER :: DP = selected_real_kind(14,200)

    interface tqsvd
        module procedure dgesvd_zero, dgesvd_one, dgesvd_two, &
                         zgesvd_zero, zgesvd_one, zgesvd_two
    end interface tqsvd

    contains

    subroutine dgesvd_prototype(A, U, S, VT, jobu, jobvt)
        implicit none
        real(DP)    , intent(in)    :: A(:,:)
        real(DP)    , intent(out)   :: U(:,:), S(:), VT(:,:)
        character   , intent(in)    :: jobu, jobvt
        !
        real(DP), allocatable       :: work(:), Acopy(:,:)
        integer                     :: m, n, info, lwork, ldvt
        !
        m = size(A, 1)
        n = size(A, 2)
        ldvt = n
        if ( jobvt .eq. 'S' ) ldvt = min(m,n)
        !
        ! let the root processor do the job
        !
        !if ( me_bgrp == root_bgrp ) then
            !
            ! workspace query
            !
            allocate( work(1), Acopy(m, n) )
            Acopy = A
            call dgesvd(jobu, jobvt, m, n, Acopy, m, S, U, m, VT, ldvt, work, -1, info)
            !
            ! allocate workspace
            !
            lwork = work(1)
            deallocate( work )
            allocate( work(lwork) )
            !
            ! perform SVD
            !
            call dgesvd(jobu, jobvt, m, n, Acopy, m, S, U, m, VT, ldvt, work, lwork, info)
            if (info .ne. 0) print *, 'dgesvd failed!'
            !
            ! clean up
            !
            deallocate( work, Acopy )
        !endif
        !
        ! let the other processors know the calculated results
        !
        !call mp_bcast( S, root_bgrp, intra_bgrp_comm )
        !if ( jobu == 'A' .or. jobu == 'S' ) call mp_bcast( U, root_bgrp, intra_bgrp_comm )
        !if ( jobvt == 'A' .or. jobvt == 'S' ) call mp_bcast( VT, root_bgrp, intra_bgrp_comm )
        !
    end subroutine dgesvd_prototype
    !
    subroutine dgesvd_zero(A, S)
        implicit none
        real(DP), intent(in)    :: A(:,:)
        real(DP), intent(out)   :: S(:)
        !
        real(DP)                :: U(0,0), VT(0,0)
        !
        call dgesvd_prototype(A, U, S, VT, 'N', 'N')
    end subroutine
    !
    subroutine dgesvd_one(A, S, X, lr, is_econ)
        implicit none
        real(DP), intent(in)            :: A(:,:)
        real(DP), intent(out)           :: S(:), X(:,:)
        logical , intent(in), optional  :: is_econ
        character, intent(in)           :: lr
        !
        character                       :: flag
        real(DP)                        :: W(0,0)
        !
        if ( (lr .ne. 'L') .and. (lr .ne. 'R') ) then
            print *, "error: dgesvd_one: the 4th argument must be 'L' or 'R'"
            return
        endif
        !
        flag = 'A'
        if (present(is_econ)) then
            if (is_econ) flag = 'S'
        endif
        !
        if (lr .eq. 'L') then
            call dgesvd_prototype(A, X, S, W, flag, 'N')
        else
            call dgesvd_prototype(A, W, S, X, 'N', flag)
        endif
    end subroutine
    !
    subroutine dgesvd_two(A, U, S, VT, is_econ)
        implicit none
        real(DP), intent(in)            :: A(:,:)
        real(DP), intent(out)           :: U(:,:), S(:), VT(:,:)
        logical , intent(in), optional  :: is_econ
        !
        character                       :: flag
        !
        flag = 'A'
        if (present(is_econ)) then
            if (is_econ) flag = 'S'
        endif
        !
        call dgesvd_prototype(A, U, S, VT, flag, flag)
    end subroutine
    !
    subroutine zgesvd_prototype(A, U, S, VT, jobu, jobvt)
        implicit none
        complex(DP) , intent(in)    :: A(:,:)
        real(DP)    , intent(out)   :: S(:)
        complex(DP) , intent(out)   :: U(:,:), VT(:,:)
        character   , intent(in)    :: jobu, jobvt
        !
        complex(DP) , allocatable   :: work(:), Acopy(:,:)
        real(DP)    , allocatable   :: rwork(:)
        integer                     :: m, n, info, lwork, ldvt
        !
        m = size(A, 1)
        n = size(A, 2)
        ldvt = n
        if ( jobvt .eq. 'S' ) ldvt = min(m,n)
        !
        ! let the root processor do the job
        !
        !if ( me_bgrp == root_bgrp ) then
            !
            ! workspace query
            !
            allocate( work(1), Acopy(m, n), rwork(5*min(m,n)) )
            Acopy = A
            call zgesvd(jobu, jobvt, m, n, Acopy, m, S, U, m, VT, ldvt, work, -1, rwork, info)
            !
            ! allocate workspace
            !
            lwork = work(1)
            deallocate( work )
            allocate( work(lwork) )
            !
            ! perform SVD
            !
            call zgesvd(jobu, jobvt, m, n, Acopy, m, S, U, m, VT, ldvt, work, lwork, rwork, info)
            if (info .ne. 0) print *, 'zgesvd failed!'
            !
            ! clean up
            !
            deallocate( work, rwork, Acopy )
        !endif
        !
        ! let the other processors know the calculated results
        !
        !call mp_bcast( S, root_bgrp, intra_bgrp_comm )
        !if ( jobu == 'A' .or. jobu == 'S' ) call mp_bcast( U, root_bgrp, intra_bgrp_comm )
        !if ( jobvt == 'A' .or. jobvt == 'S' ) call mp_bcast( VT, root_bgrp, intra_bgrp_comm )
        !
    end subroutine zgesvd_prototype
    !
    subroutine zgesvd_zero(A, S)
        implicit none
        complex(DP) , intent(in)    :: A(:,:)
        real(DP)    , intent(out)   :: S(:)
        !
        complex(DP)                 :: U(0,0), VT(0,0)
        !
        call zgesvd_prototype(A, U, S, VT, 'N', 'N')
    end subroutine
    !
    subroutine zgesvd_one(A, S, X, lr, is_econ)
        implicit none
        complex(DP) , intent(in)            :: A(:,:)
        real(DP)    , intent(out)           :: S(:)
        complex(DP) , intent(out)           :: X(:,:)
        logical     , intent(in), optional  :: is_econ
        character, intent(in)               :: lr
        !
        character                           :: flag
        complex(DP)                         :: W(0,0)
        !
        if ( (lr .ne. 'L') .and. (lr .ne. 'R') ) then
            print *, "error: zgesvd_one: the 4th argument must be 'L' or 'R'"
            return
        endif
        !
        flag = 'A'
        if (present(is_econ)) then
            if (is_econ) flag = 'S'
        endif
        !
        if (lr .eq. 'L') then
            call zgesvd_prototype(A, X, S, W, flag, 'N')
        else
            call zgesvd_prototype(A, W, S, X, 'N', flag)
        endif
    end subroutine
    !
    subroutine zgesvd_two(A, U, S, VT, is_econ)
        implicit none
        complex(DP) , intent(in)            :: A(:,:)
        real(DP)    , intent(out)           :: S(:)
        complex(DP) , intent(out)           :: U(:,:), VT(:,:)
        logical     , intent(in), optional  :: is_econ
        !
        character                           :: flag
        !
        flag = 'A'
        if (present(is_econ)) then
            if (is_econ) flag = 'S'
        endif
        !
        call zgesvd_prototype(A, U, S, VT, flag, flag)
    end subroutine
    !


end module svd
