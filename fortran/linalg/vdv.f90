program test
    implicit none

    integer :: nt, sz, i, j, it
    integer :: t1, t2, rate

    real(kind=8), allocatable :: V(:,:), B(:,:), ev(:), X(:,:), Y(:,:), tmp(:)
    real(kind=8), external :: ddot

    nt = 10
    sz = 500

    allocate(V(sz,sz), B(sz,sz), ev(sz), X(sz,sz), Y(sz,sz), tmp(sz))

    call random_number(V)
    call random_number(B)
    call random_number(ev)


    call system_clock(t1, rate)
    do it=1,nt

        X = 0.d0

        do i=1,sz
            tmp = V(i,:) / sqrt(ev)
            !X(i,i)= dot_product(V(i,:), tmp)
            X(i,i) = ddot(sz, V(i,1), sz, tmp, 1)
            do j = i+1,sz
                !X(i,j) = dot_product(V(j,:),tmp)
                X(i,j) = ddot(sz, V(j,1), sz, tmp, 1)
                X(j,i) = X(i,j)
            enddo
        enddo

    enddo
    call system_clock(t2, rate)
    print *, 'dumb: ', real(t2-t1)/rate

    !do i=1,sz
    !    print *, X(i,:)
    !enddo
    !print *, ''

    call system_clock(t1, rate)
    do it = 1,nt

        Y = 0.d0

        do i=1,sz
            call dsyr('u', sz, 1.d0/sqrt(ev(i)), V(:,i), 1, Y, sz)
        enddo

    enddo
    call system_clock(t2, rate)
    print *, 'dsyr: ', real(t2-t1)/rate

    print *, abs(X(10,200)-Y(10,200))

    !do i=1,sz
    !    print *, Y(i,:)
    !enddo
    !print *, ''

    !
    !B = X-Y
    !do i=1,sz
    !    print *, B(i,:)
    !enddo

end program test
