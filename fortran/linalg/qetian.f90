module qetian
    implicit none

contains

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
        call dposv('U', sz, 1, Acopy, sz, x, sz, info)
        if (info .ne. 0) print *, 'dposv failed.'
        !
        deallocate(Acopy)
    end subroutine tqsolve

    SUBROUTINE tqzshexp(expA, A, alpha)
    	IMPLICIT NONE
    	COMPLEX(kind=8), INTENT(IN)  :: A(:,:)
    	COMPLEX(kind=8)              :: expA(:,:)
    	REAL(kind=8)                 :: alpha
    	COMPLEX(kind=8), ALLOCATABLE :: iA(:,:), tmpvec(:)
    	INTEGER                  :: sz
    	!
    	COMPLEX(kind=8), ALLOCATABLE :: eigvec(:,:), work(:)
    	REAL(kind=8),    ALLOCATABLE :: eigval(:), rwork(:)
    	INTEGER,     ALLOCATABLE :: iwork(:)
    	INTEGER                  :: i, j, lwork, lrwork, liwork, info
    	!
    	sz = SIZE(A,1)
    	IF ( sz .NE. SIZE(A,2) .OR. sz .NE. SIZE(expA,1) .OR. sz .NE. SIZE(expA,2) ) THEN
    		PRINT *, 'Invalid size when calling tqzshexp'
    		RETURN
    	ENDIF
    	ALLOCATE( iA(sz,sz) )
    	iA = alpha * (0.d0,1.d0) * A
    	!
    	! workspace query
    	ALLOCATE( work(1) )
    	ALLOCATE( rwork(1) )
    	ALLOCATE( iwork(1) )
    	ALLOCATE( eigval(sz) )
    	ALLOCATE( eigvec(sz,sz) )
    	CALL zheevd('V','U',sz,eigvec,sz,eigval,work,-1,rwork,-1,iwork,-1,info)
    	!
    	! allocate workspace
    	lwork = work(1)
    	lrwork = rwork(1)
    	liwork = iwork(1)
        print *, 'lwork = ', lwork
        print *, 'lrwork = ', lrwork
        print *, 'liwork = ', liwork
    	DEALLOCATE( work )
    	DEALLOCATE( rwork )
    	DEALLOCATE( iwork )
    	ALLOCATE( work(lwork) )
    	ALLOCATE( rwork(lrwork) )
    	ALLOCATE( iwork(liwork) )
    	!
    	! diagonalization
    	eigvec = iA
    	CALL zheevd('V','U',sz,eigvec,sz,eigval,work,lwork,rwork,lrwork,iwork,liwork,info)
    	DEALLOCATE( work )
    	DEALLOCATE( rwork )
    	DEALLOCATE( iwork )
    	IF (ABS(info) .GT. 0) THEN
    		PRINT *, "Diagonalization failed in tqzshexp"
    		DEALLOCATE(eigval)
    		DEALLOCATE(eigvec)
    		RETURN
    	ENDIF
    	!
    	! calculate expA = V * diag(exp(-iD)) * V'
    	ALLOCATE( tmpvec(sz) )
    	DO i = 1, sz
    		tmpvec = eigvec(i,:) * EXP(-(0.d0,1.d0)*eigval)
    		DO j = 1, sz
    			expA(i,j) = DOT_PRODUCT(eigvec(j,:), tmpvec)
    		ENDDO
    	ENDDO
    	!
    	DEALLOCATE(eigval)
    	DEALLOCATE(eigvec)
    		DEALLOCATE(tmpvec)
    END SUBROUTINE tqzshexp

end module
