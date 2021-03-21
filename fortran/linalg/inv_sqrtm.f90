program main
    implicit none

    real(kind=8), allocatable :: A(:,:)
    real(kind=8), allocatable :: B(:,:)
    real(kind=8), allocatable :: Q(:,:)
    real(kind=8), allocatable :: L(:)

    integer :: i,j,k,sz=3

    !interface
    !    subroutine inv_sqrtm(A, B)
    !        real(kind=8), allocatable, intent(in) :: A(:,:)
    !        real(kind=8), allocatable :: B(:,:)
    !    end subroutine inv_sqrtm
    !end interface
    allocate(A(sz, sz))
    allocate(Q(sz, sz))
    allocate(L(sz))

    Q(:,1) = (/ 0.379973078879873d0,-0.885348235614359d0, 0.267915958877203d0 /)
    Q(:,2) = (/-0.059009803536167d0, 0.265848488357072d0, 0.962207058966459d0 /)
    Q(:,3) = (/ 0.923113374625876d0, 0.381422446812980d0,-0.048771043191900d0 /)
    L = (/ 1.d0, 2.d0, 3.d0 /)

    A(:,:) = 0.d0
    do i = 1, sz
        do j = 1, sz
            A(i,j) = sum(Q(i,:)*L*Q(j,:))
        enddo
    enddo

    !write(*,*) 'Q=', Q
    write(*,*) 'A=', A

    allocate(B(sz,sz))

    !call inv_sqrtm2(sz, A, B)
    call invsqrtm_sypd(A, B)

    write(*,*) 'B=', B

    deallocate(A)
    deallocate(B)


contains

    !---------------------------------------------------------------------
    subroutine invsqrtm_sypd(A, B)
    !---------------------------------------------------------------------
        !! calculate the matrix inverse square root
        !! of a real symmetric positive-definite matrix
        !
        !! Given a real symmetric positive-definite matrix A
        !! B = A^(-1/2)
        !
        implicit none
        !
        real(kind=8) :: A(:,:)
        real(kind=8) :: B(:,:)
        !
        real(kind=8), allocatable ::  eigvec(:,:), eigval(:), tmpvec(:), work(:)
        integer, allocatable :: iwork(:)
        integer :: n, i, j, lwork = -1, liwork = -1, info
        !
        n = size(A,1)
        !
        ! workspace query
        !
        allocate( work(1) )
        allocate( iwork(1) )
        allocate( eigval(n) )
        allocate( eigvec(n,n) )
        call dsyevd('V', 'U', n, eigvec, n, eigval, work, lwork, iwork, liwork, info)
        !
        ! allocate workspace
        !
        lwork = work(1)
        liwork = iwork(1)
        deallocate( work )
        deallocate( iwork )
        allocate( work(lwork) )
        allocate( iwork(liwork) )
        !
        ! diagonalization                                                                 
        !                                                                                 
        eigvec = A                                                                        
        call dsyevd('V', 'U', n, eigvec, n, eigval, work, lwork, iwork, liwork, info)     
        !                                                                                 
        ! deallocate workspace                                                            
        !                                                                                 
        deallocate( work )                                                                
        deallocate( iwork )                                                               
        !                                                                                 
        ! sanity check                                                                    
        !                                                                                 
        print *, 'info = ', info
        !                                                                                 
        if ( any(eigval < epsilon(eigval)) ) info = 1                                     
        print *, 'info = ', info
        !                                                                                 
        ! calculate B = V * diag(D.^(-1/2)) * V'                                          
        !                                                                                 
        allocate( tmpvec(n) )                                                             
        do i = 1, n                                                                       
            tmpvec = eigvec(i,:) / dsqrt(eigval)                                          
            do j = i, n                                                                   
                B(i,j) = dot_product(tmpvec, eigvec(j,:))                                 
                if (i .ne. j) B(j,i) = B(i,j)                                             
            enddo                                                                         
        enddo                                                                             
        !                                                                                 
        ! clear temporary variables                                                       
        !                                                                                 
        deallocate(eigval)                                                                
        deallocate(eigvec)                                                                
        deallocate(tmpvec)                                                                
        !                                                                                 
    end subroutine invsqrtm_sypd                                                              
    
!subroutine inv_sqrtm2(sz, A, B)
!    implicit none
!    integer :: sz, lwork = -1, liwork = -1, info, i, j
!    real(kind=8) :: A(sz,sz)
!    real(kind=8) :: B(sz,sz)
!
!    real(kind=8), allocatable :: eigvec(:,:), eigval(:)
!    real(kind=8), allocatable :: work(:)
!    integer, allocatable :: iwork(:)
!
!    allocate(eigvec(sz,sz))
!    allocate(eigval(sz))
!    allocate(work(1))
!    allocate(iwork(1))
!
!    eigvec = A
!    call dsyevd('V', 'U', sz, eigvec, sz, eigval, work, lwork, iwork, liwork, info)
!
!    lwork = work(1)
!    liwork = iwork(1)
!    deallocate(work)
!    deallocate(iwork)
!    allocate(work(lwork))
!    allocate(iwork(liwork))
!
!    call dsyevd('V', 'U', sz, eigvec, sz, eigval, work, lwork, iwork, liwork, info)
!    deallocate(work)
!    deallocate(iwork)
!
!    write(*,*) 'eigvec = ', eigvec
!    write(*,*) 'eigval = ', eigval
!
!    B = matmul(eigvec/spread(dsqrt(eigval),1,sz), transpose(eigvec))
!    !do i = 1, sz
!    !    do j = i, sz
!    !        B(i,j) = sum(eigvec(i,:) / dsqrt(eigval) * eigvec(j,:))
!    !        B(j,i) = B(i,j)
!    !    enddo
!    !enddo
!
!end subroutine inv_sqrtm2

!subroutine inv_sqrtm(A, B)
!    implicit none
!    real(kind=8), allocatable, intent(in) :: A(:,:)
!    real(kind=8), allocatable :: B(:,:)
!    real(kind=8), allocatable :: eigvec(:,:), eigval(:), work(:)
!    integer :: sz, lwork = -1, liwork = -1, info, i, j, k
!    integer, allocatable :: iwork(:)
!
!    sz = size(A,1)
!    write(*,*) 'sz = ', sz
!    write(*,*) 'inv_sqrtm: A = ', A 
!    allocate(eigval(sz))
!    allocate(eigvec(sz,sz))
!    allocate(work(1))
!    allocate(iwork(1))
!
!    eigvec = A
!    call dsyevd('V', 'U', sz, eigvec, sz, eigval, work, lwork, iwork, liwork, info)
!    write(*,*) 'info = ', info
!
!    lwork = work(1)
!    liwork = iwork(1)
!    deallocate(work)
!    deallocate(iwork)
!    allocate(work(lwork))
!    allocate(iwork(liwork))
!
!    call dsyevd('V', 'U', sz, eigvec, sz, eigval, work, lwork, iwork, liwork, info)
!    deallocate(work)
!    deallocate(iwork)
!
!    write(*,*) 'eigvec = ', eigvec
!    write(*,*) 'eigval = ', eigval
!
!    if (.not. allocated(B)) allocate(B(sz,sz))
!    if ((size(B,1) /= sz .or. size(B,2) /= sz )) then
!        deallocate(B)
!        allocate(B(sz,sz))
!    endif
!    B(:,:) = 0.d0
!    do i = 1, sz
!        do j = i, sz
!            B(i,j) = sum(eigvec(i,:) / dsqrt(eigval) * eigvec(j,:))
!            !do k = 1, sz
!            !    B(i,j) = B(i,j) + eigvec(i,k) / dsqrt(eigval(k)) * eigvec(j,k)
!            !enddo
!            if (j /= i) B(j,i) = B(i,j)
!        enddo
!    enddo
!
!    deallocate(eigval)
!    deallocate(eigvec)
!
!end subroutine inv_sqrtm


end program main
