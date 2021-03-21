program test
    include 'mpif.h'
    real(kind=8) :: t0, t1, dt
    integer :: id, nproc, ierr
    !
    call mpi_init(ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, id, ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nproc, ierr)
    !
    print *, 'id = ', id
    print *, 'nproc = ', nproc
    !
    t0 = mpi_wtime()
    call sleep(2)
    t1 = mpi_wtime()
    dt = t1-t0
    print *, 't0 = ', t0
    print *, 't1 = ', t1
    print *, 'dt = ', dt
    !
    call mpi_finalize(ierr)
end program
