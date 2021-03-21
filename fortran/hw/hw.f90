module test

contains
    subroutine pm(tar, n1, n2, n3)
        implicit none
        integer, pointer :: ptr(:)
        integer, allocatable, target, intent(in) :: tar(:)
        integer :: n1, n2, n3, ntot

        ntot = n1*n2*n3

        ptr => tar

        print *, permute(ptr)

        nullify(ptr)

        contains
            function permute(par) result(ppar)
                implicit none
                integer, pointer, intent(in) :: par(:)
                integer, allocatable :: ppar(:)
                allocate( ppar(ntot) )
                ppar = reshape(reshape(par, [n3, n2, n1], order=[3,2,1]), [ntot])
            end function
    end subroutine pm
end module test

program main 
    use test
    implicit none
    integer, allocatable :: iarr(:,:,:), ia(:)
    integer :: sz1, sz2,sz3,i,j,k

    sz1 = 2
    sz2 = 3
    sz3 = 4
    allocate(iarr(sz1,sz2,sz3), ia(sz1*sz2*sz3))

    do i=1,sz1
        do j = 1,sz2
            do k=1,sz3
                iarr(i,j,k) = 100*i+10*j+k
            enddo
        enddo
    enddo

    ia = reshape(iarr,[sz1*sz2*sz3])
    print *,ia 
    call pm(ia, sz1, sz2, sz3)

    deallocate(iarr, ia)

    
end program main

