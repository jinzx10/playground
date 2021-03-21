module lsmod
    implicit none
    real(kind=8) :: alpha

contains

    function func(alpha) result(f)
        implicit none
        real(kind=8), intent(in) :: alpha
        real(kind=8) :: f
        f = 0.5d0*(alpha-0.8d0)**2
    end function

    function dfunc(alpha) result(df)
        implicit none
        real(kind=8), intent(in) :: alpha
        real(kind=8) :: df
        real(kind=8) :: dalpha
        dalpha = 1.0d-4
        df = ( func(alpha+dalpha) - func(alpha-dalpha) ) / 2.d0 / dalpha
    end function

    subroutine linesearch(flag)
        implicit none
        real(kind=8) :: c1, c2
        real(kind=8), allocatable :: fvals(:), dvals(:), avals(:)
        integer :: max_iter_ls, i, flag

        flag = 1
        max_iter_ls = 20

        ! parameters for the strong Wolfe conditions
        c1 = 0.25d0 ! sufficient decrease
        c2 = 0.5d0 ! slope

        allocate( fvals(max_iter_ls) )
        allocate( dvals(max_iter_ls) )
        allocate( avals(max_iter_ls) )

        dvals(1) = dfunc(0.d0)
        if ( dvals(1) .ge. 0 ) then
            print *, 'line search direction is not descent.'
        endif

        fvals(1) = func(0.d0)
        avals(1) = 0.d0
        avals(2) = 1.d0

        do i = 2, max_iter_ls
            fvals(i) = func(avals(i))
            if ( ( fvals(i) .gt. (fvals(1)+c1*avals(i)*dvals(1)) ) &
                .or. ( fvals(i) .ge. fvals(i-1) ) ) then
                call zoom(avals(i-1), avals(i), c1, c2, fvals(1), dvals(1))
                flag = 0
                exit
            endif

            dvals(i) = dfunc(avals(i))
            if ( abs(dvals(i)) .le. -c2*dvals(1) ) then
                alpha = avals(i)
                flag = 0
                exit
            endif

            if ( dvals(i) .ge. 0.d0 ) then
                call zoom(avals(i), avals(i-1), c1, c2, fvals(1), dvals(1))
                flag = 0
                exit
            endif

            if (i .ne. max_iter_ls) avals(i+1) = 2.d0 * avals(i)
        enddo

        deallocate(fvals)
        deallocate(dvals)
        deallocate(avals)
    end subroutine linesearch

    subroutine zoom(alpha1, alpha2, c1, c2, ftmp0, dtmp0)
        implicit none
        real(kind=8), intent(in) :: alpha1, alpha2, c1, c2, ftmp0, dtmp0
        integer ::  max_iter_zoom, j
        real(kind=8) :: alpha_low, alpha_high, ftmp, dtmp

        max_iter_zoom = 50
        alpha_low = alpha1
        alpha_high = alpha2

        do j = 1, max_iter_zoom
            alpha = 0.5d0*(alpha_low+alpha_high)
            ftmp = func(alpha);
            if ( ( ftmp .gt. (ftmp0 + c1*alpha*dtmp0) ) & 
                .or. ( ftmp .ge. func(alpha_low) ) ) then
                alpha_high = alpha
            else
                dtmp = dfunc(alpha)
                if ( abs(dtmp) <= -c2*dtmp0 ) then
                    return
                endif
                if ( dtmp*(alpha_high-alpha_low) .ge. 0 ) then
                    alpha_high = alpha_low;
                endif
                alpha_low = alpha;
            endif
        enddo
        
        print *, 'zoom failed!'

    end subroutine zoom


end module


