program test
    use mmmod
    implicit none

    integer, parameter :: dp = selected_real_kind(15,200)
    real(dp), allocatable :: A(:,:), B(:,:), C(:,:)
    integer :: sz, nt, rate, i
    integer :: td_start, td_end, dd
    integer :: ts_start, ts_end, ds
    integer :: tb_start, tb_end, db

    print *, 'enter matrix size: '
    read *, sz
    print *, 'enter the number of trials: '
    read *, nt

    allocate( A(sz,sz), B(sz,sz), C(sz-1,sz-1) )
    A = 1.d0
    B = 0.5d0
    C = 0.d0

    ! block 
    dd = 0
    ds = 0
    db = 0
    do i = 1, nt
        call system_clock(td_start, rate)
        call dgemm('N', 'N', sz-1, sz-1, sz-1, 1.d0, A, sz, B, sz, 0.d0, C, sz-1)
        call system_clock(td_end, rate)
        dd = dd + td_end - td_start

        call system_clock(tb_start, rate)
        call dgemm('N', 'N', sz-1, sz-1, sz-1, 1.d0, A(1:sz-1,1:sz-1), sz-1, B(1:sz-1,1:sz-1), sz-1, 0.d0, C, sz-1)
        call system_clock(tb_end, rate)
        db = db + tb_end - tb_start

        call system_clock(ts_start, rate)
        call mm(A(1:sz-1,1:sz-1), B(1:sz-1,1:sz-1), C)
        call system_clock(ts_end, rate)
        ds = ds + ts_end - ts_start
    enddo

    write(*,*) 'direct dgemm: average elapsed time = ', real(dd)/rate/nt, ' seconds'
    write(*,*) 'block pass  : average elapsed time = ', real(db)/rate/nt, ' seconds'
    write(*,*) 'subroutine  : average elapsed time = ', real(ds)/rate/nt, ' seconds'

end program


