module overload_test
    implicit none
    real(kind=8), parameter :: e = dexp(1.d0)

    interface mysum
        module procedure dsum
        module procedure isum
        !subroutine dsum(darr, res)
        !    implicit none
        !    real(kind=8) :: darr(:), res
        !end subroutine dsum
        !subroutine isum(iarr, res)
        !    implicit none
        !    integer :: iarr(:), res
        !end subroutine isum
    end interface mysum

contains

    subroutine dsum(darr, res)
        implicit none
        real(kind=8) :: darr(:), res
        res = sum(darr) 
    end subroutine dsum

    subroutine isum(iarr, res)
        implicit none
        integer :: iarr(:), res
        res = sum(iarr) 
    end subroutine isum

    subroutine copy(A,B)
        implicit none
        real(kind=8) :: A(:), B(:)
        B = A
    end subroutine copy


    !subroutine prototype(A, B, C, s)
    !    implicit none
    !    integer :: A(:), B(:), C(:), s
    !    s = sum(A) + sum(B) + sum(C) 
    !end subroutine prototype

    !subroutine instance(A, B, s)
    !    implicit none
    !    integer :: A(:), B(:), C(:), s
    !    call prototype(A, B, C, s)
    !end subroutine

    subroutine sub()
        implicit none
        call subsub()
        contains 
            subroutine subsub()
                implicit none
                print *, 'subsub!'
            end subroutine
    end subroutine
end module overload_test
