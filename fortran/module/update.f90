module update
    implicit none
    real(kind=8) :: a

contains
    subroutine test
        implicit none
        integer :: i = 0
        i = i + 1
        print *, 'i = ', i
    end subroutine

    subroutine init()
        implicit none
        a = 3.14d0
    end subroutine init

    subroutine toz()
        implicit none
        a = 0
    end subroutine toz

    subroutine printnadd()
        implicit none
        integer :: ii

        print *, 'a = ', a
        print *, 'ii = ', ii

        call addone()
        call addone()
        call restart()

    contains

        subroutine addone()
            implicit none
            a = a + 1.d0
            ii = ii + 1.d0
        end subroutine addone

        subroutine restart()
            implicit none
            call toz()
        end subroutine

    end subroutine printnadd
end module 
