module hw
    implicit none
contains

    function outprod(x,y) result(z)
        implicit none
        real, intent(in) :: x(:), y(:)
        real, dimension(size(x), size(y)) :: z
        z = matmul( reshape(x,(/size(x),1/)), reshape(y,(/1,size(y)/)) )
    end function
end module hw

program test
    use hw
    implicit none
    real :: z(3,3), x(3), y(2)
    integer :: i

    x = 1.
    y = 2.
    z = outprod(x,y)

    do i = 1,size(z,1)
        print *,z(i,:)
    enddo

    


end program test

