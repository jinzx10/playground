module samemod

contains

    subroutine test(zin, zout)
        real(kind=8) :: zin(:,:), zout(:,:)
        zout = 2.d0 * zin
    end subroutine
end module samemod
