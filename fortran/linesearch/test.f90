program test
    implicit none
    real(kind=8) :: c1, c2, alpha
    integer :: flag
    c1 = 0.25d0
    c2 = 0.5d0
    call tq_line_search(func, dfunc, c1, c2, alpha, flag)
    print *, 'alpha = ', alpha
    print *, 'flag = ', flag 

contains

    function func(alpha) result(f)
        implicit none
        real(kind=8), intent(in) :: alpha
        real(kind=8) :: f
        f = 0.5d0*(alpha-0.1d0)**2
        !f = 0.5d0 * (1.d0-exp(-2.d0*(alpha-0.1d0)))**2
    end function
    
    function dfunc(alpha) result(df)
        implicit none
        real(kind=8), intent(in) :: alpha
        real(kind=8) :: df
        real(kind=8) :: dalpha
        dalpha = 1.0d-4
        df = ( func(alpha+dalpha) - func(alpha-dalpha) ) / 2.d0 / dalpha
    end function

end program

! inexact line search (Nocedal p.60-61)
subroutine tq_line_search(func, dfunc, wolfe1, wolfe2, alpha, flag)
    implicit none
    !
    ! parameters for the strong Wolfe conditions
    ! wolfe1: sufficient decrease
    ! wolfe2: curvature 
    real(kind=8), intent(in) :: wolfe1, wolfe2
    !
    ! line search result
    real(kind=8) :: alpha
    !
    ! iteration history
    real(kind=8), allocatable :: fvals(:), dvals(:), avals(:)
    !
    ! maximum number of iterations
    integer :: max_iter
    !
    ! on exit, if flag equals
    ! 0 --> success
    ! 1 --> search direction is not descent
    ! 2 --> failed within the given number of iterations
    ! 3 --> zoom failed
    integer :: flag
    !
    ! loop variable
    integer :: i
    !
    interface
        function func(alpha)
            real(kind=8), intent(in) :: alpha
            real(kind=8) :: func
        end function
        function dfunc(alpha)
            real(kind=8), intent(in) :: alpha
            real(kind=8) :: dfunc
        end function
    end interface
    !
    max_iter = 20
    alpha = 1.d0
    flag = 0
    !
    allocate( fvals(max_iter) )
    allocate( dvals(max_iter) )
    allocate( avals(max_iter) )
    !
    dvals(1) = dfunc(0.d0)
    if ( dvals(1) .ge. 0 ) then
        flag = 1
        return
    endif
    !
    fvals(1) = func(0.d0)
    avals(1) = 0.d0
    avals(2) = 1.d0
    !
    do i = 2, max_iter
        fvals(i) = func(avals(i))
        if ( ( fvals(i) .gt. (fvals(1)+wolfe1*avals(i)*dvals(1)) ) &
            .or. ( fvals(i) .ge. fvals(i-1) ) ) then
            call zoom(avals(i-1), avals(i))
            exit
        endif
        !
        dvals(i) = dfunc(avals(i))
        if ( abs(dvals(i)) .le. -wolfe2*dvals(1) ) then
            alpha = avals(i)
            exit
        endif
        !
        if ( dvals(i) .ge. 0.d0 ) then
            call zoom(avals(i), avals(i-1))
            exit
        endif
        !
        if (i .ne. max_iter) then
            avals(i+1) = 2.d0 * avals(i)
        else
            flag = 2
        endif
    enddo
    !
    select case(flag)
        case(0) 
            !print *, 'line search finished.' 
        case(1) 
            print *, 'line search direction is not descent.' 
        case(2) 
            print *, 'line search failed within ', max_iter, ' iterations.'
        case(3) 
            print *, 'zoom failed.' 
        case default
            print *, 'something goes wrong...'
    end select
    !
    deallocate(fvals)
    deallocate(dvals)
    deallocate(avals)
    !
contains
    !
    subroutine zoom(alpha1, alpha2)
        implicit none
        real(kind=8), intent(in) :: alpha1, alpha2
        integer ::  max_iter_zoom, j
        real(kind=8) :: alpha_low, alpha_high
        real(kind=8) :: ftmp, dtmp
        !
        max_iter_zoom = 40
        alpha_low = alpha1
        alpha_high = alpha2
        !
        do j = 1, max_iter_zoom
            alpha = 0.5d0*(alpha_low+alpha_high)
            ftmp = func(alpha);
            if ( ( ftmp .gt. (fvals(1) + wolfe1*alpha*dvals(1)) ) & 
                .or. ( ftmp .ge. func(alpha_low) ) ) then
                alpha_high = alpha
            else
                dtmp = dfunc(alpha)
                if ( abs(dtmp) <= -wolfe2*dvals(1) ) then
                    return
                endif
                if ( dtmp*(alpha_high-alpha_low) .ge. 0 ) then
                    alpha_high = alpha_low;
                endif
                alpha_low = alpha;
            endif
        enddo
        !    
        flag = 3
        !
    end subroutine zoom
    !
end subroutine tq_line_search

