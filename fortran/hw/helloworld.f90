module helloworld 

    implicit none
contains

    subroutine test_init()
        integer :: i = 1
        i = i + 1
        print *, 'i = ', i
    end subroutine

!    recursive subroutine factorial(n, nfact)
!        integer :: n, nfact
!    
!        if (n==1) then
!            nfact = 1
!        else
!            call factorial(n-1, nfact)
!            nfact = n*nfact
!        endif
!    
!    end subroutine factorial
!
!    recursive function ffact(n) result(res)
!        integer :: n, res
!
!        if (n==1) then 
!            res = 1
!        else
!            res = n * ffact(n-1)
!        endif
!
!    end function ffact
!
!
!    function diff1(func, x0) result(df)
!        real(kind=8) :: x0, fp, fm, dx = 1.0e-3, df
!
!        interface
!            function func(x)
!                real(kind=8) :: x
!                real(kind=8) :: func
!            end function func
!        end interface
!
!        fp = func(x0+dx)
!        fm = func(x0-dx)
!        df = (fp-fm)/2.0/dx
!
!    end function diff1
!
!    function sqr(x) result(res)
!        real(kind=8) :: x, res
!
!        res = x*x
!    end function sqr
!
!    subroutine newtonroot(f, x, beta, max_iter, tol, flag)
!        real(kind=8) :: x, beta, tol, fx, delta, J, dx, fx_new
!        integer :: max_iter, flag, counter
!
!        interface 
!            function f(y)
!                real(kind=8) :: y
!                real(kind=8) :: f
!            end function f
!        end interface
!
!        fx = f(x)
!        if (abs(fx) < tol) then
!            flag = 0
!            return
!        endif
!
!        delta =  1.0d-3 * max(1.0d0, sqrt(abs(x)))
!        J = (f(x+delta) - fx) / delta
!
!        dx = 0.0d0
!        fx_new = 0.0d0
!
!        do counter = 1, max_iter
!            if (abs(J) < 1.0d-14) then
!                print *, 'newtonroot: the Jacobian appears to be singular.'
!                flag = 2
!                return
!            endif
!
!            dx = -fx / J * beta
!            x = x + dx
!            fx_new = f(x)
!            if (abs(fx_new) < tol) then
!                flag = 0
!                return
!            endif
!
!            J = (fx_new - fx) / dx
!            fx = fx_new
!        enddo
!
!        print *, 'newtonroot: fails to find the root.'
!    end subroutine newtonroot
!
!    ! stolen from http://fortranwiki.org/fortran/show/Matrix+inversion
!    function inv(A) result(Ainv)
!        real(kind=8), dimension(:,:), intent(in) :: A
!        real(kind=8), dimension(size(A,1),size(A,2)) :: Ainv
!        
!        real(kind=8), dimension(size(A,1)) :: work  ! work array for LAPACK
!        integer, dimension(size(A,1)) :: ipiv   ! pivot indices
!        integer :: n, info
!        
!        ! External procedures defined in LAPACK
!        external dgetrf
!        external dgetri
!        
!        ! Store A in Ainv to prevent it from being overwritten by LAPACK
!        Ainv = A
!        n = size(A,1)
!        
!        ! DGETRF computes an LU factorization of a general M-by-N matrix A
!        ! using partial pivoting with row interchanges.
!        call dgetrf(n, n, Ainv, n, ipiv, info)
!        
!        if (info /= 0) then
!           stop 'Matrix is numerically singular!'
!        end if
!        
!        ! DGETRI computes the inverse of a matrix using the LU factorization
!        ! computed by DGETRF.
!        call dgetri(n, Ainv, n, ipiv, work, n, info)
!        
!        if (info /= 0) then
!           stop 'Matrix inversion failed!'
!        end if
!    end function inv
!
!    subroutine broydenroot(f, x, beta, max_iter, tol, flag)
!        real(kind=8), intent(in) :: beta, tol
!        integer, intent(in) :: max_iter
!        real(kind=8), dimension(:) :: x
!
!        real(kind=8), dimension(size(x)) :: fx, dxi, df, dx, fx_new
!        real(kind=8), dimension(size(x), size(x)) :: J, invJ
!        real(kind=8) :: p(size(x),1), q(1,size(x))
!        real(kind=8) :: delta
!        integer :: flag, i, counter
!
!        interface
!            function f(y)
!                real(kind=8), dimension(:), intent(in) :: y
!                real(kind=8), dimension(size(y)) :: f
!            end function f
!        end interface
!
!        delta = 1.d-6 * max(1.d0, sqrt(maxval(abs(x))))
!        fx = f(x)
!
!        ! initial Jacobian
!        do i = 1, size(x)
!            dxi = 0.d0
!            dxi(i) = delta
!            df = f(x+dxi) - fx
!            J(:,i) = df / delta
!        enddo
!
!        invJ = inv(J)
!        !print *, 'initial Jacobian', invJ
!
!        ! loop
!        do counter = 1, max_iter
!            dx = -matmul(invJ, fx) * beta
!            print *, 'dx = ', dx
!            x = x + dx
!            print *, 'x = ', x
!            fx_new = f(x)
!            print *, 'fx_new = ', fx_new
!            if (maxval(abs(fx_new)) < tol) then
!                flag = 0
!                print *, 'root found', x
!                return
!            endif
!
!            df = fx_new - fx
!            print *, 'df = ', df
!            fx = fx_new
!            print *, 'fx = ', fx
!            p(:,1) = ( dx - matmul(invJ,df) ) / sum(dx*matmul(invJ,df))
!            print *, 'p = ', p
!            q(1,:) = matmul(dx,invJ)
!            print *, 'q = ', q
!            invJ = invJ + matmul(p,q)
!            print *, 'invJ = ', invJ
!        enddo
!
!        print *, 'broydenroot: fails to find the root'
!        flag = 1
!
!    end subroutine broydenroot

end module helloworld

