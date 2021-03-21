program hw2
    implicit none
    real(kind=8),allocatable :: A(:,:), B(:,:), C(:,:)
    integer :: i

    allocate(A(2,2),B(5,5))
    A = 1.d0
    B = 2.d0

    print *, 'B = '
    do i = 1, size(B,1)
        print *, B(i,:)
    enddo

    B = A

    print *, 'B = '
    do i = 1, size(B,1)
        print *, B(i,:)
    enddo



    deallocate(A,B)
    
end program
