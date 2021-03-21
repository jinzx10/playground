program hw
    implicit none
    real(kind=8) :: x(4)
    call init_random_seed

    x = 0.d0
    print *, 'x = ', x
    call random_number(x(2))
    print *, 'x = ', x
end program

SUBROUTINE init_random_seed()
    IMPLICIT NONE
    INTEGER :: i, n, clock
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    
    print *, 'n = ', n
    print *, 'i = ', i

    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    
    CALL SYSTEM_CLOCK(COUNT=clock)
    
    seed = clock + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    
    print *, 'n = ', n
    print *, 'seed = ', seed
    DEALLOCATE(seed)
END SUBROUTINE
