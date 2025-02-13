!===============================================================================
!
! Copyright (C) 2020 Institute of Theoretical Astrophysics, University
! of Oslo.
!
! This file is part of Commander3.
!
! Commander3 is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! Commander3 is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with Commander3. If not, see <https://www.gnu.org/licenses/>.
!
!===============================================================================

! This module handles corrections to time ordered data due to adc issues
submodule (comm_tod_adc_mod) comm_tod_adc_smod
contains

  !=========================================================================
  ! This code is based off of the DPC ADC corrections (thanks Bob).
  ! 
  ! Those corrections procede as such:
  !
  ! The detector has some characteristic response function V' = R(V)V
  ! which is assumed to be near unity everywhere.
  !
  ! As the sky signal changes, the change in the measured detector
  ! voltage is given by 
  ! 
  !             \deltaV' = (V * dR(V)/dV + R(V) )* \deltaV
  !
  ! We aim to find the derivative dR(V)/dV of the response function to 
  ! correct for the non-linearties described in 
  ! Planck 2013: III https://arxiv.org/abs/1303.5064
  !
  ! DCH
  !=========================================================================

  module function constructor_internal(cpar, info, nbins)
    ! ====================================================================
    ! Sets up an adc correction object that maps input and output voltages
    ! Also initializes the bins used for the actual correction model
    !
    ! Inputs:
    ! 
    ! comm   : integer
    !          mpi communicator
    !
    ! myid   : integer
    !          mpi identifier
    !
    ! nbins  : integer
    !          number of bins used for building the adc correction tables
    !
    ! Returns:
    ! --------
    !
    ! constructor_internal: pointer
    !    contains all of the bins needed for computing adc corrections
    !    and the actual correction tables
    ! ====================================================================
    
    implicit none
    integer(i4b),           intent(in) :: nbins
    class(comm_mapinfo),    target     :: info
    class(comm_adc),        pointer    :: constructor_internal
    type(comm_params),      intent(in) :: cpar
    
    real(sp)     :: diff

    integer(i4b) :: i, ierr
    
    allocate(constructor_internal)
    
    constructor_internal%window  =  10
    constructor_internal%info    => info
    constructor_internal%myid    =  cpar%myid_chain
    constructor_internal%comm    =  cpar%comm_chain
    constructor_internal%outdir  =  cpar%outdir
    constructor_internal%nbins   =  nbins
    
    allocate(constructor_internal%adc_in(constructor_internal%nbins), constructor_internal%adc_out(constructor_internal%nbins))
    allocate(constructor_internal%rms_bins(constructor_internal%nbins), constructor_internal%v_bins(constructor_internal%nbins))
    allocate(constructor_internal%rms2_bins(constructor_internal%nbins))
    allocate(constructor_internal%nval(constructor_internal%nbins), constructor_internal%vbin_edges(constructor_internal%nbins+1))
    allocate(constructor_internal%err_bins(constructor_internal%nbins))

    ! For the corrected stuffs
    allocate(constructor_internal%rms_bins2(constructor_internal%nbins))
    allocate(constructor_internal%nval2(constructor_internal%nbins))

    constructor_internal%rms_bins2(:)  = 0.0
    constructor_internal%nval2(:)      = 0

    constructor_internal%adc_in(:)     = 0.0
    constructor_internal%adc_out(:)    = 0.0
    constructor_internal%vbin_edges(:) = 0.0
    constructor_internal%v_bins(:)     = 0.0
    constructor_internal%rms_bins(:)   = 0.0
    constructor_internal%rms2_bins(:)  = 0.0
    constructor_internal%nval(:)       = 0
    constructor_internal%err_bins(:)   = 0.0

    ! ! Initialize v_min and v_max on obscenely wrong numbers
    constructor_internal%v_max = 0.0
    constructor_internal%v_min = 100000.0

  end function constructor_internal

  module function constructor_precomp(instfile, path, load)
    ! ====================================================================
    ! Sets up an adc correction object that maps input and output voltages
    ! Also initializes the bins used for the actual correction model
    !
    ! Inputs:
    ! 
    ! comm   : integer
    !          mpi communicator
    !
    ! myid   : integer
    !          mpi identifier
    !
    ! nbins  : integer
    !          number of bins used for building the adc correction tables
    !
    ! Returns:
    ! --------
    !
    ! constructor: pointer
    !    contains all of the bins needed for computing adc corrections
    !    and the actual correction tables
    ! ====================================================================
    
    implicit none
    type(hdf_file),     intent(in) :: instfile
    character(len=512), intent(in) :: path
    logical(lgt),       intent(in) :: load
    class(comm_adc),    pointer    :: constructor_precomp
    
    integer(i4b) :: ext(2), col, i, mingood, maxgood
    real(dp), dimension(:,:), allocatable :: buffer
    real(dp), dimension(:),   allocatable :: buffer_in, buffer_out

    allocate(constructor_precomp)
        
    ! read in adc correction templates
    call get_size_hdf(instfile, path, ext)
    allocate(buffer(ext(1), ext(2)))
    call read_hdf(instfile, path, buffer)
    col = 1; if (load) col = 3
    do while (buffer(ext(1),col) == 0.d0)
       ext(1) = ext(1)-1
    end do

    allocate(buffer_in(ext(1)))
    allocate(buffer_out(ext(1)))

    buffer_in  = buffer(1:ext(1),col+1)
    buffer_out = buffer(1:ext(1),col)

    constructor_precomp%nbins  = 500
    constructor_precomp%window = 10

    allocate(constructor_precomp%rms_bins2(500))
    allocate(constructor_precomp%nval2(500))
    allocate(constructor_precomp%v_bins(500))
    allocate(constructor_precomp%vbin_edges(501))

    ! Find DPC min and max voltage values to grid out the voltages well
    ! We assume that voltage in and voltage out share edges 
    do i = 1, ext(1)
       mingood = i
       if (buffer_in(i) > 0.0) exit
    end do
    do i = ext(1), 1, -1
       maxgood = i
       if (buffer_in(i) /= 0.0 .and. buffer_in(i) /= 10.0) exit
    end do

    allocate(constructor_precomp%adc_in(maxgood-mingood))
    allocate(constructor_precomp%adc_out(maxgood-mingood))

    constructor_precomp%adc_in  = buffer(1:ext(1),col+1)
    constructor_precomp%adc_out = buffer(1:ext(1),col)
    constructor_precomp%v_min   = constructor_precomp%adc_in(mingood)
    constructor_precomp%v_max   = constructor_precomp%adc_in(maxgood)

    constructor_precomp%nval2(:)      = 0
    constructor_precomp%rms_bins2(:) = 0.0
    deallocate(buffer)
    call spline(constructor_precomp%sadc, real(constructor_precomp%adc_in,dp), real(constructor_precomp%adc_out,dp))

  end function constructor_precomp

  module subroutine adc_correct(self, tod_in, tod_out, scan, det, di)
    !=========================================================================
    ! Adc corrects a timestream 
    ! 
    ! Inputs:
    !
    ! self : comm_adc object
    !    Defines the adc correction that should be applied
    ! tod_in : float array
    !    The tod that is to be corrected
    ! scan   : integer
    !    tod scan number
    ! det    : integer
    !    instrument detector identifier
    ! di     : integer
    !    detector diode identifier
    ! 
    ! Outputs : 
    !
    ! tod_out : float array
    !    The adc corrected version of tod_in    
    ! ====================================================================
    implicit none
    class(comm_adc),                 intent(inout) :: self
    real(sp), dimension(:),          intent(in)    :: tod_in
    real(sp), dimension(:),          intent(out)   :: tod_out
    integer(i4b), intent(in), optional :: scan, det, di

    integer(i4b)                                   :: i

    ! allocate(self%sadc(leng))
    do i = 1, size(tod_in)
       if (tod_in(i) < self%v_min .or. tod_in(i) > self%v_max) then
          tod_out(i) = tod_in(i)
       else
          tod_out(i) = splint(self%sadc,real(tod_in(i),dp))
       end if
       !if (abs(tod_in(i)-tod_out(i))/tod_in(i) > 1d-2) then
       !    write(*,*) scan, det, di, tod_in(i), tod_out(i), (tod_in(i)-tod_out(i))/tod_in(i)
       !end if
    end do
    
  end subroutine adc_correct

  module subroutine build_table(self,handle,name)
    !=========================================================================
    ! Adc corrects a timestream by fitting regular Gaussian dips to the binned
    ! RMS values of the 
    ! 
    ! Inputs:
    !
    ! self:     comm_adc object
    !           Defines the adc correction that should be applied
    ! handle:   type(planck_rng)
    !           Healpix random number type
    ! name:     string
    !           diode name for output file names
    ! Outputs: 
    !
    ! volt_in:  float array
    !           Array of the input voltages
    !
    ! volt_out: float
    !           array of the corrected voltages
    implicit none
    class(comm_adc),                 intent(inout) :: self
    type(planck_rng),                intent(inout) :: handle
    character(len=50),               intent(in)    :: name
    integer(i4b), dimension(:),      allocatable   :: binmask
    integer(i4b), dimension(:),      allocatable   :: dummymask
    integer(i4b), dimension(:),      allocatable   :: v_dips
    integer(i4b)                                   :: i, j, leng
    integer(i4b)                                   :: ierr, trims
    integer(i4b)                                   :: dip1, v_off, diprange
    real(sp)                                       :: sum, slope, offset

    real(dp),     dimension(:),      allocatable   :: lin_dp, term1, term2
    real(dp),     dimension(:),      allocatable   :: rms_dp, flat_dp, model_dp
    real(dp),     dimension(:),      allocatable   :: vbin_dp, dRdV, R
    real(dp)                                       :: slope_dp, offset_dp
    real(dp)                                       :: a, zero, m, delV
    real(dp)                                       :: m1, b, alpha

    logical(lgt) :: steamroll, bad

    steamroll = .false.!.true.

    ! Combine together all of the bins determined from chunk adding
    call mpi_allreduce(mpi_in_place,self%rms_bins,self%nbins,MPI_REAL, MPI_SUM, self%comm, ierr)
    call mpi_allreduce(mpi_in_place,self%rms2_bins,self%nbins,MPI_REAL, MPI_SUM, self%comm, ierr)
    call mpi_allreduce(mpi_in_place,self%nval,self%nbins,MPI_INTEGER, MPI_SUM, self%comm, ierr)
    
    ! The rest should be light enough to do on a single core
    if (self%myid == 0) then
       ! open(44, file=trim(self%outdir)//'/adc_WNsum_'//trim(name)//'.dat')
       ! open(45, file=trim(self%outdir)//'/adc_WNsum2_'//trim(name)//'.dat')
       ! open(46, file=trim(self%outdir)//'/adc_WNn_'//trim(name)//'.dat')
       ! open(47, file=trim(self%outdir)//'/adc_vb_'//trim(name)//'.dat')
       ! do i = 1, self%nbins
       !    write(44, fmt='(e30.8)') self%rms_bins(i)
       !    write(45, fmt='(e30.8)') self%rms2_bins(i)
       !    write(46, fmt='(i9)')    self%nval(i)
       !    write(47, fmt='(e30.8)') self%v_bins(i)
       ! end do
       ! close(44)
       ! close(45)
       ! close(46)
       ! close(47)

       ! Allocate and intialize everything
       allocate(binmask(self%nbins))
       allocate(dummymask(self%nbins))
       allocate(vbin_dp(self%nbins), rms_dp(self%nbins))
       allocate(dRdV(self%nbins),R(self%nbins))
       allocate(model_dp(self%nbins))
       allocate(lin_dp(self%nbins))
       allocate(flat_dp(self%nbins))

       allocate(term1(self%nbins))
       allocate(term2(self%nbins))

       binmask(:)   = 1
       dummymask(:) = 1

       ! Mask bad bins and massive outliers       
       call mask_bins(self%v_bins, self%rms_bins, self%nval, binmask)

       vbin_dp(:) = real(self%v_bins,dp)
       rms_dp(:)  = real(self%rms_bins,dp)

       ! Remove the linear term from V vs RMS before indentifying the dips
       call return_linreg_dp(vbin_dp, rms_dp, binmask, slope_dp, offset_dp, trim=.true.)

       flat_dp = rms_dp - slope_dp*vbin_dp + offset_dp

       ! Remove linear term 
       if (steamroll) then
          ! write(*,*) 'Steamroll correction'
          bad = .false.
          do i = 1, self%nbins
             if (binmask(i) == 0) cycle
             model_dp(i) = -flat_dp(i)
          end do
       else
          ! If a negative slope, don't fit dips
          if (slope_dp < 0.d0) bad = .true.

          ! How large of a bin range do we think the dips will cover?
          diprange = 10

          ! Identify dip locations for the fitting search
          call return_dips_dp(vbin_dp, flat_dp, binmask, diprange, v_dips, name)

          ! Fit linear and dips jointly
          bad = .not. allocated(v_dips)

          ! If we look at one of these diodes, don't even try
          if (trim(name) == '18S_sky10') bad = .true.
          if (trim(name) == '22S_ref10') bad = .true.
          if (trim(name) == '22S_sky11') bad = .true.
          if (trim(name) == '22S_ref11') bad = .true.
          if (trim(name) == '22S_sky10') bad = .true.
          if (trim(name) == '24S_ref10') bad = .true.
          if (trim(name) == '24S_sky10') bad = .true.
          if (trim(name) == '26S_ref10') bad = .true.
          if (trim(name) == '26S_sky10') bad = .true.
          
          if (.not. bad) then
             model_dp = return_gauss_lin_model_dp(vbin_dp, rms_dp, binmask, self%nval, slope_dp, offset_dp, v_dips, handle, name) 
          else
             model_dp = 0.d0
          end if
       end if

       if (bad) then
          self%adc_in  = vbin_dp
          self%adc_out = vbin_dp
          dRdV    = 0.
          R       = 0.
          lin_dp  = slope_dp*vbin_dp + offset_dp
       else
          lin_dp  = slope_dp*vbin_dp + offset_dp
          flat_dp = rms_dp - lin_dp
          ! Taking our model, we return the differential response function (dR/dV)
          if (all(model_dp /= 0)) then
             do i = 1, self%nbins
                dRdV(i)  = slope_dp/model_dp(i) - 1.0/vbin_dp(i) 
                term1(i) = slope_dp/model_dp(i)
                term2(i) = 1.0/vbin_dp(i)
             end do
             model_dp = model_dp + offset_dp
          else
             dRdV = 0.
          end if
          
          ! Make sure the bottom of dRdV is flat (dRdV(0) = dRdV(nbins) = 0.0)
          dRdV = dRdV - (dRdV(self%nbins)-dRdV(1))/(vbin_dp(self%nbins)-vbin_dp(1))*vbin_dp
          dRdV = dRdV - minval(dRdV)
          ! Actual inverse response function
          R(1) = 1.0
          delV = vbin_dp(2)-vbin_dp(1)
          do i = 2, self%nbins
             R(i) = dRdV(i)*delV + R(i-1)
          end do
          
          ! The following is done to match the dpc tables
          
          ! Again make sure everything is nice and flat
          m = (R(self%nbins)-R(1))/(vbin_dp(self%nbins)-vbin_dp(1))
          
          do i = 1, self%nbins
             R(i) = R(i)-m*vbin_dp(i)
          end do
          
          ! And then make the limits 0
          zero = R(1)
          do i = 1, self%nbins
             R(i) = R(i) - zero
             self%adc_out(i) = R(i)*vbin_dp(i)+vbin_dp(i)
             self%adc_in(i)  = vbin_dp(i)
          end do
          
       end if
       ! ! Write to file binned rms, voltages, and response function to files
       ! open(44, file=trim(self%outdir)//'/adc_binned_rms_'//trim(name)//'_flat.dat')
       ! open(45, file=trim(self%outdir)//'/adc_linear_term_'//trim(name)//'.dat')
       ! open(46, file=trim(self%outdir)//'/adc_response_function_'//trim(name)//'.dat')
       ! open(47, file=trim(self%outdir)//'/adc_term1_'//trim(name)//'.dat')
       ! open(48, file=trim(self%outdir)//'/adc_term2_'//trim(name)//'.dat')
       ! open(49, file=trim(self%outdir)//'/adc_model_'//trim(name)//'.dat') 
       ! open(50, file=trim(self%outdir)//'/adc_binned_rms_'//trim(name)//'.dat')
       ! ! open(52, file=trim(self%outdir)//'/adc_in_'//trim(name)//'.dat') 
       ! ! open(53, file=trim(self%outdir)//'/adc_out_'//trim(name)//'.dat') 
       ! open(54, file=trim(self%outdir)//'/adc_binmask_'//trim(name)//'.dat') 
       ! open(55, file=trim(self%outdir)//'/adc_dRdV_'//trim(name)//'.dat') 
       ! do i = 1, self%nbins
       !    write(44, fmt='(e30.8)') flat_dp(i)
       !    write(45, fmt='(e16.8)') lin_dp(i)
       !    write(46, fmt='(e16.8)') R(i)
       !    write(47, fmt='(e16.8)') term1(i)
       !    write(48, fmt='(e16.8)') term2(i)
       !    write(49, fmt='(e16.8)') model_dp(i)
       !    write(50, fmt='(e16.8)') rms_dp(i)
       !    ! write(52, fmt='(e16.8)') self%adc_in(i)
       !    ! write(53, fmt='(e16.8)') self%adc_out(i)
       !    write(54, fmt='(i1)')    binmask(i)
       !    write(55, fmt='(e16.8)') dRdV(i)
       ! end do
       ! close(44)
       ! close(45)
       ! close(46)
       ! close(47)
       ! close(48)
       ! close(49)
       ! close(50)
       ! ! close(52)
       ! ! close(53)
       ! close(54)
       ! close(55)
       
       deallocate(binmask, dummymask, vbin_dp, rms_dp, dRdV, R, model_dp, lin_dp, flat_dp)
          
    end if

    ! mpi_bcast the tables to all other cores
    call mpi_bcast(self%adc_in,  self%nbins, MPI_REAL, 0, self%comm, ierr) 
    call mpi_bcast(self%adc_out, self%nbins, MPI_REAL, 0, self%comm, ierr) 

    ! call spline(self%sadc, real(self%adc_in,dp), real(self%adc_out,dp), regular=.true.)
    call spline(self%sadc, real(self%adc_in,dp), real(self%adc_out,dp))
    
  end subroutine build_table

  module subroutine construct_voltage_bins(self)

    implicit none
    class(comm_adc), intent(inout) :: self
    integer(i4b)                   :: i, ierr

    if (self%myid == 0) then

       ! Declare bin edges
       do i = 1, self%nbins+1
          self%vbin_edges(i) = (self%v_max-self%v_min)*(i-1)/self%nbins + self%v_min
       end do
       ! Declare bins
       do i = 1, self%nbins
          self%v_bins(i) = (self%vbin_edges(i) + self%vbin_edges(i+1))/2.0
       end do
    end if
    
    call mpi_bcast(self%vbin_edges,self%nbins, MPI_REAL, 0, self%comm, ierr)
    call mpi_bcast(self%v_bins,    self%nbins, MPI_REAL, 0, self%comm, ierr)

  end subroutine construct_voltage_bins
  
  module subroutine find_horn_min_max(self,tod_in,flag,flag0)
    ! ==================================================================
    ! This subroutine loops through the diode data to find the global
    ! minimum and maximum voltages
    !
    ! Inputs:
    !
    ! self     : comm_adc object
    !    Defines the adc correction that should be applied
    !
    ! tod_in   : float array (sp)
    !    The chunk of data which we will be looping over
    ! 
    ! flag     : integer array
    !    data flagging corresponding to the chunk of data 
    !
    ! flag0    : integer
    !    something I don't really know what it does
    ! 
    ! "Outputs":
    !
    ! self%v_min : float (sp)
    !    updated global minimum voltage
    !
    ! self%v_max : float (sp)
    !    updated global maximum voltage
    !
    ! ====================================================================
    implicit none
    class(comm_adc),                   intent(inout) :: self
    real(sp),     dimension(:),        intent(in)    :: tod_in
    integer(i4b), dimension(:),        intent(in)    :: flag
    integer(i4b),                      intent(in)    :: flag0

    integer(i4b)                                     :: i, ierr, leng

    leng = size(tod_in)

    do i = 1, leng
       if (iand(flag(i),flag0) .ne. 0) cycle 
       if (tod_in(i) < self%v_min) self%v_min = tod_in(i)
       if (tod_in(i) > self%v_max) self%v_max = tod_in(i)
    end do

  end subroutine find_horn_min_max

  module subroutine bin_scan_rms(self,tod_in,flag,flag0,corr)
    ! ====================================================================
    ! This subroutine takes in a chunk of data (from wherever it lives) and puts
    ! it in the appropriate bins
    !
    ! Inputs:
    !
    ! self     : comm_adc object
    !    Defines the adc correction that should be applied
    !
    ! tod_in   : float array (sp)
    !    The chunk of data which we will be binning here
    !
    !
    ! "Outputs":
    !
    !   self%nval: integer array
    !     Counts the number of entries in each bin
    ! 
    !   self%binval: float array (sp)
    !     Sum of all rms values for each bin
    ! ====================================================================
    
    implicit none
    class(comm_adc),               intent(inout) :: self
    real(sp),     dimension(:),    intent(in)    :: tod_in
    integer(i4b), dimension(:),    intent(in)    :: flag
    integer(i4b),                  intent(in)    :: flag0

    logical(lgt),      optional,   intent(in)    :: corr ! follow slightly different procedure if data is corrected

    real(sp),          dimension(:), allocatable :: dV, tod_trim
    real(sp)                                     :: sum, binwidth
    integer(i4b)                                 :: leng, i, j, j_min, j_max
    character(len=4) :: myid

    call int2string(self%myid, myid)
    
    leng = size(tod_in)
    
    binwidth = self%vbin_edges(2) - self%vbin_edges(1)

    allocate(dV(leng))
    dV = 1e30

    ! Compute the dV within a window around each rt sample (excluding the ends)
    do i = 1, leng
       if (iand(flag(i),flag0) .ne. 0) cycle 
       sum = 0.d0
       j_min = max(i-int(self%window/2),1)
       j_max = min(i+int(self%window/2),leng-1)
       do j = j_min, j_max
          if (iand(flag(j),flag0) .ne. 0 .or. iand(flag(j+1),flag0) .ne. 0) cycle 
          sum = sum + (tod_in(j+1)-tod_in(j))**2
       end do
       dV(i) = sqrt(sum/(j_max-j_min+1))
    end do
    
    ! Bin the dV values as a function of input voltage, and take the mean
    if (present(corr)) then
       do i = 1, leng-1
          if (iand(flag(i),flag0) .ne. 0) cycle 
          j = int((tod_in(i)-self%vbin_edges(1))/binwidth) + 1
          if (j > self%nbins) cycle
          self%nval2(j)     = self%nval2(j) + 1
          self%rms_bins2(j) = self%rms_bins2(j) + dV(i)
       end do
    else
       do i = 1, leng-1
          if (iand(flag(i),flag0) .ne. 0) cycle 
          j = int((tod_in(i)-self%vbin_edges(1))/binwidth) + 1
          if (j > self%nbins) cycle
          self%nval(j)      = self%nval(j) + 1
          self%rms_bins(j)  = self%rms_bins(j)  + dV(i)
          self%rms2_bins(j) = self%rms2_bins(j) + dV(i)**2
       end do
    end if

    deallocate(dV)
    
  end subroutine bin_scan_rms

  module subroutine corr_rms_out(self,name)
    ! ====================================================================
    !
    ! Test function to output the rms of the corrected TODs
    !
    ! Input: 
    !
    ! self : class
    !        comm_adc
    ! name : character
    !        diode name
    ! ====================================================================
    implicit none
    class(comm_adc),   intent(inout) :: self
    character(len=50), intent(in)    :: name
    integer(i4b)                     :: i, ierr

    call mpi_allreduce(mpi_in_place,self%rms_bins2,self%nbins,MPI_REAL, MPI_SUM, self%comm, ierr)
    call mpi_allreduce(mpi_in_place,self%nval2,self%nbins,MPI_INTEGER, MPI_SUM, self%comm, ierr)

    if (self%myid == 0) then
       open(27,file=trim(self%outdir)//'/'//trim(name)//'_corrected_rms.dat')
       open(28,file=trim(self%outdir)//'/'//trim(name)//'_corrected_nval.dat')
       open(29,file=trim(self%outdir)//'/'//trim(name)//'_corrected_vbins.dat')
       do i = 1, self%nbins
          write(27,*) self%rms_bins2(i)
          write(28,*) self%nval2(i)
          write(29,*) self%v_bins(i)
       end do
       close(27)
       close(28)
       close(29)
    end if

  end subroutine corr_rms_out

  module subroutine mask_bins(vbins,rms,nval,mask)
    ! ====================================================================
    ! This subroutine iterates masks out bins with spuriously large deviations
    ! in the white noise level (spikes) and bins with 0 entries
    !
    ! Inputs:
    !
    ! vbins:     float array
    !            the 'x-axis': the voltage bins
    !
    ! rms:       float array
    !            the 'y-axis': the rms bins
    !
    ! nval:      integer array
    !            the count of entries for each rms bin
    !
    ! mask:      integer array
    !            binary array which determines which bins contribute to the fitting procedure
    !
    ! Outputs:
    !
    ! mask:      integer array
    !            binary array which determines which bins contribute to the fitting procedure
    ! ====================================================================
    implicit none
    real(sp),     dimension(:), intent(in)    :: vbins
    integer(i4b), dimension(:), intent(in)    :: nval
    real(sp),     dimension(:), intent(inout) :: rms
    integer(i4b), dimension(:), intent(inout) :: mask
    real(sp)                                  :: m, b
    integer(i4b)                              :: i, j, k, leng, count
    real(sp),     dimension(:), allocatable   :: rms_flat
    real(sp)                                  :: nval_mean, y_mean, y_std, y_var

    ! Initialize the middle mean and std
    leng = size(vbins)
    allocate(rms_flat(leng))

    nval_mean = 0.0
    y_mean    = 0.0
    y_std     = 0.0
    y_var     = 0.0

    do i = 1, leng
       nval_mean = nval_mean + nval(i)
    end do
    nval_mean = nval_mean/leng

    ! Mask out bins that have no entries - otherwise return mean rms for each bin
    do i = 1, leng
       if (nval(i) == 0) then
          mask(i) = 0
          cycle
       end if
       if (rms(i) == 0) then
          mask(i) = 0
          cycle
       end if
       mask(i) = 1
       rms(i)  = rms(i)/nval(i)       
    end do

    ! Mask out under sampled bins too
    do i = 1, leng
       if (mask(i) == 0) cycle
       if (nval(i) < 0.1*nval_mean) then
          mask(i) = 0
       end if
    end do

    ! Fit the linear portion for the currently unmasked bins
    call return_linreg_sp(vbins,rms,mask,m,b,trim=.true.)

    rms_flat = rms - m*vbins - b

    ! Find the mean and standard deviation to filter out spikey behavior
    count = 0
    do i = 1, leng
       if (mask(i) == 0) cycle
       count  = count + 1
       y_mean = y_mean + rms_flat(i)
    end do
    if (count > 0) y_mean = y_mean/count
    
    count = 0
    do i = 1, leng
       if (mask(i) == 0) cycle
       count = count + 1
       y_var  = y_var + (rms_flat(i)-y_mean)**2
    end do
    
    if (count > 1) y_var = y_var/(count-1)
    
    y_std  = sqrt(y_var)

    do i = 1, leng
       if (mask(i) == 0) cycle
       if (abs(rms_flat(i) - rms_flat(i-1)) > y_std .and. abs(rms_flat(i+1) - rms_flat(i)) > y_std) then
          mask(i) = 0
       end if
    end do
    
  end subroutine mask_bins
    
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!! DP FUNCTIONS !!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


  module subroutine return_linreg_dp(x,y,mask,slope,offset,trim)
    !=========================================================================
    ! Very simple function that fits the slope and offset for an x-y pair
    !
    ! Inputs:
    !
    ! x      : float array (sp)
    !          The independent variable
    ! y      : float array (sp)
    !          The dependent variable
    ! mask   : integer array
    !          which array values do not contribute to the fit? 
    ! trim   : logical (optional)
    !          are large deviations disregarded in the linear fit?
    !
    ! Output:
    !
    ! slope  : float (sp)
    !          The slope of the linear fit
    ! offset : float (sp)
    !          The offset of the linear fit
    !=========================================================================    
    implicit none
    
    real(dp),     dimension(:),  intent(in)    :: x, y
    integer(i4b), dimension(:),  intent(in)    :: mask
    logical(lgt), optional,      intent(in)    :: trim
    real(dp),                    intent(inout) :: slope, offset
    real(dp), dimension(:,:),    allocatable   :: adata, bdata
    real(dp), dimension(:),      allocatable   :: work
    real(dp), dimension(:),      allocatable   :: x2, y2
    integer(i4b)                               :: i, leng, count
    real(dp)                                   :: y_mean, y_var, y_std
    integer(i4b)                               :: info
    
    leng = size(x)

    y_mean = 0.0
    y_var  = 0.0
    y_std  = 0.0

    count = 0
    do i = 1, leng
       if (mask(i) == 0) cycle
       count = count + 1
       y_mean = y_mean + y(i)
    end do
    if (count == 0) then
       slope = 0.d0
       offset = 0.d0
       return
    end if
    y_mean = y_mean/count
    
    do i = 1, leng
       if (mask(i) == 0) cycle
       y_var  = y_var + (y(i)-y_mean)**2
    end do
    
    y_var = y_var/(count-1)
    
    y_std  = sqrt(y_var)

    allocate(x2(leng),y2(leng))

    x2(:) = 0.0
    y2(:) = 0.0

    ! Mask out bins that have no entries
    count = 0
    do i = 1, leng
       ! Mask out outliers (if (y < mean-std .or. y > mean + std))
       if (present(trim)) then
          if (y(i) < y_mean-2.0*y_std) cycle
          if (y(i) > y_mean+2.0*y_std) cycle
       end if
       if (mask(i) == 0) cycle
       count     = count + 1
       x2(count) = x(i)
       y2(count) = y(i) 
    end do

    allocate(adata(count,2), bdata(count,1), work(2*count))
    adata(:,:) = 0.0
    bdata(:,:) = 0.0
    work(:)    = 0.0
    
    adata(:,1) = 1.0
    adata(:,2) = x2(1:count)
    bdata(:,1) = y2(1:count)
    
    call dgels('N', count, 2, 1, adata, count, bdata, count, work, 2*count, info)
    
    slope  = bdata(2,1)
    offset = bdata(1,1)
    
    deallocate(adata)
    deallocate(bdata)
    deallocate(work)
    
  end subroutine return_linreg_dp

  module subroutine return_dips_dp(x,y,mask,diprange,res,name)
    ! ====================================================================
    ! Dip identifying subroutine that returns the first dip location and
    ! the distance between dips in the RMS
    !
    ! Inputs:
    !
    ! x       : float array
    !            input voltage axis
    ! 
    ! y       : float array
    !           binned rms array with the linear part removed
    !
    ! mask    : integer array
    !           mask for binned rms array - 0 if y(i) = NaN or has been trimmed
    !
    ! diprange: integer
    !           how many x axis indices we want to search for dips within
    !
    ! Outputs:
    !
    ! dips    : integer array
    !           array of the dip locations in the voltage bin array
    ! ====================================================================
    implicit none
    
    real(dp),     dimension(:), intent(in)    :: x, y
    integer(i4b), dimension(:), intent(inout) :: mask
    integer(i4b),               intent(in)    :: diprange
    character(len=50),          intent(in)    :: name
    logical(lgt), dimension(:), allocatable   :: truths
    integer(i4b)                              :: loc(1)
    integer(i4b), dimension(:), allocatable   :: dips
    integer(i4b), dimension(:), allocatable   :: res
    real(dp),     dimension(:), allocatable   :: manual_dips

    integer(i4b)                              :: leng, i, j, count
    integer(i4b)                              :: ndips
    real(dp)                                  :: y_mean, y_var, y_std
    
    leng = size(x)
    
    allocate(truths(diprange))
    allocate(dips(leng))

    dips(:)   = 0
    ndips     = 0
    truths(:) = .false.
    y_mean    = 0.0
    y_var     = 0.0

    ! Determine mean and standard deviation of the input y-array
    count = 0
    do i = 1, leng
       if (mask(i) == 0) cycle
       count  = count + 1
       y_mean = y_mean + y(i)
    end do
    if (count == 0) then
       deallocate(truths, dips)
       return
    end if
    y_mean = y_mean/count
    
    count = 0
    do i = 1, leng
       if (mask(i) == 0) cycle
       count = count + 1
       y_var  = y_var + (y(i)-y_mean)**2
    end do
    
    y_var = y_var/(count-1)
    
    y_std  = sqrt(y_var)

    ! Since the linear portion has been removed, mean should be near zero,
    ! so dips are identified first by finding y-values where y < -1.0*y_std

    if (trim(name) == '19S_ref10') then
       ndips = 6
       allocate(manual_dips(ndips))
       manual_dips = (/0.8922, 0.8992, 0.9064, 0.9133, 0.9198, 0.9268 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '19S_sky10') then
       ndips = 6
       allocate(manual_dips(ndips))
       manual_dips = (/0.8433, 0.8504, 0.8572, 0.8643, 0.8711, 0.8780 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '19S_ref11') then
       ndips = 9
       allocate(manual_dips(ndips))
       manual_dips = (/1.1056, 1.1127, 1.1195, 1.1265, 1.1403, 1.1477, 1.1539, 1.1617, 1.1678 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '19S_sky11') then
       ndips = 10
       allocate(manual_dips(ndips))
       manual_dips = (/1.0638, 1.0706, 1.0775, 1.0847, 1.0913, 1.0985, 1.1055, 1.1124, 1.1197, 1.1263 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do

    else if (trim(name) == '21S_ref10') then
       ndips = 9
       allocate(manual_dips(ndips))
       manual_dips = (/1.0022, 1.0093, 1.0160, 1.0233, 1.0304, 1.0373, 1.0445, 1.0513, 1.0585 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '21S_sky10') then
       ndips = 9
       allocate(manual_dips(ndips))
       manual_dips = (/0.9607, 0.9676, 0.9745, 0.9815, 0.9884, 0.9953, 1.0023, 1.0092, 1.0166 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '21S_ref11') then
       ndips = 12
       allocate(manual_dips(ndips))
       manual_dips = (/1.0063, 1.0128, 1.0204, 1.0269, 1.0410, 1.0482, 1.0549, 1.0621, 1.0693, 1.0764, 1.0833, 1.0899 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '21S_sky11') then
       ndips = 12
       allocate(manual_dips(ndips))
       manual_dips = (/0.9850, 0.9921, 0.9988, 1.0062, 1.0132, 1.0197, 1.0270, 1.0351, 1.0410, 1.0479, 1.0555, 1.0625 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do

    else if (trim(name) == '22M_ref01') then
       ndips = 7
       allocate(manual_dips(ndips))
       manual_dips = (/0.6347,0.6415,0.6488,0.6557,0.6644,0.6717,0.6785/)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '22M_sky01') then
       ndips = 7
       allocate(manual_dips(ndips))
       manual_dips = (/0.6203,0.6271,0.6344,0.6415,0.6502,0.6573,0.6643/)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do

    else if (trim(name) == '23S_ref10') then
       ndips = 15
       allocate(manual_dips(ndips))
       manual_dips = (/1.1758, 1.1829, 1.1897, 1.1966, 1.2038, 1.2108, 1.2179, 1.2249, 1.2319, 1.2386, 1.2457, 1.2529, 1.2600, 1.2669, 1.2739 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '23S_sky10') then
       ndips = 15
       allocate(manual_dips(ndips))
       manual_dips = (/1.1266, 1.1336, 1.1406, 1.1477, 1.1547, 1.1616, 1.1685, 1.1757, 1.18276, 1.1897, 1.1967, 1.2038, 1.2108, 1.2180, 1.225 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '23S_ref11') then
       ndips = 8
       allocate(manual_dips(ndips))
       manual_dips = (/0.6199, 0.6269, 0.6340, 0.6410, 0.6480, 0.6550, 0.6616, 0.6687 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else if (trim(name) == '23S_sky11') then
       ndips = 8
       allocate(manual_dips(ndips))
       manual_dips = (/0.5988, 0.6058, 0.6128, 0.6198, 0.6268, 0.6338, 0.6408, 0.6478 /)
       do i = 1, ndips
          loc     =  minloc(abs(x-manual_dips(i)),DIM=1)
          dips(i) = loc(1)
       end do
    else
       ! Don't allow dips along the edges
       do i = 10, leng-10
          if (mask(i) == 0) cycle
          ! Only consider variations
          if (y(i) < y_mean-2.0*y_std .and. y(i-1) < y_mean-2.0*y_std .and. y(i+1) < y_mean-2.0*y_std) then
             truths(:) = .false.
             ! search local range
             do j = 1, diprange
                if (i+j == leng) then
                   truths(j:) = .true.
                   exit
                end if
                ! if lower than your neighbors, share the good news!
                if (y(i) < y(i-j) .and. y(i) < y(i+j)) then
                   truths(j) = .true.
                else
                   truths(j) = .false.
                end if
             end do
             ! If lower than all your neighbors
             if (all(truths)) then
                ! append a dip location
                ndips       = ndips + 1
                dips(ndips) = i
             end if
          end if
       end do
    end if
    if (ndips .gt. 0) then
       allocate(res(ndips))
       res = dips(1:ndips)
    end if

    deallocate(truths)
    deallocate(dips)

  end subroutine return_dips_dp

  module function return_gaussian_dp(x, pars) result(y)
    ! ====================================================================
    ! Super simple function which returns a guassian function given the parameters
    !
    ! Inputs: 
    !
    ! x     : float array
    !         the x-array over which the gaussian is evaluated
    ! pars  : float array (size = 3)
    !         mean, stddev, and amplitude of the gaussian
    !
    ! Outputs:
    !
    ! y     : float array
    !         returned gaussian function
    ! ====================================================================
    
    implicit none
    
    real(dp), dimension(:), intent(in)         :: x
    real(dp), dimension(3), intent(in)         :: pars
    real(dp), allocatable, dimension(:)        :: y
    real(dp)                                   :: mu, sigma, amp
    integer(i4b)                               :: leng, i

    leng = size(x)
    
    allocate(y(leng))

    mu    = pars(1)
    sigma = pars(2)
    amp   = pars(3)

    do i = 1, leng
       y(i) = amp * exp(-((x(i)-mu)/sigma)**2)
    end do
    
  end function return_gaussian_dp

  module function return_gauss_lin_model_dp(x, y, mask, bincount, a, b, dips, handle, name) result(model)
    !=========================================================================
    ! Fits a gaussian function to each recognized dip in the 
    ! white noise level
    !
    ! Inputs:
    ! 
    ! x:       float array
    !          voltage bins from the tod_in binning
    ! y:       float array
    !          rms bins from the rms estimates as a function of voltage
    ! mask:    integer array
    !          mask for binned rms array - 0 if y(i) = NaN or has been trimmed
    ! dips:    integer array
    !          index value for the location of the dips to be fit
    ! handle:  type(planck_rng)
    !          Healpix random number type
    !
    ! Outputs: 
    !
    ! idrf:    float array 
    !          array of length(x) - the inverse differential response function
    !          which will be integrated
    !=========================================================================
    
    implicit none
    
    real(dp),     dimension(:), intent(in)    :: x, y
    real(dp),                   intent(inout) :: a, b ! slope and offset of the model fit
    integer(i4b), dimension(:), intent(in)    :: mask
    integer(i4b), dimension(:), intent(in)    :: bincount
    integer(i4b), dimension(:), intent(in)    :: dips
    type(planck_rng),           intent(inout) :: handle
    real(dp),     dimension(:), allocatable   :: newy, model
    real(dp)                                  :: sigma, amp, mean, fwhm
    real(dp)                                  :: sigma_est, amp_est, mean_est
    integer(i4b)                              :: leng, i, j, ndips
    integer(i4b)                              :: fit_range

    real(dp)                                  :: a2, a_old

    character(len=3)                          :: gibbstr, lstr
    character(len=2)                          :: dip_str
    character(len=50),          intent(in)    :: name

    real(dp),     dimension(:), allocatable   :: x_tmp, y_tmp, count_tmp
    integer(i4b), dimension(:), allocatable   :: mask_tmp, dummymask

    real(dp),     dimension(:), allocatable   :: gdips, dipless, gmodel
    real(dp),     dimension(:), allocatable   :: flaty

    integer(i4b)                              :: currpar, first, last, ngibbs, k, l

    real(dp), dimension(2)                    :: P_uni
    real(dp), dimension(3)                    :: pars
    real(dp), dimension(3)                    :: par_est
    
    ! declare the goodies
    fit_range = 30
    ngibbs    = 25
    leng      = size(x)
    
    ! allocate all relevant arrays
    allocate(newy(leng))
    allocate(model(leng))

    allocate(gdips(leng))
    allocate(dummymask(leng))
    allocate(dipless(leng))
    allocate(flaty(leng))
    allocate(gmodel(leng))

    ! init cumulative arrays
    newy(:)     = 0.d0
    gdips(:)    = 0.d0
    flaty(:)    = 0.d0

    dummymask(:) = 1

    ndips = size(dips)

    ! For each iteration we take the previous slope and gaussian fits 
    do l = 1, ngibbs
       a_old = a

       ! Given the gaussian fits, find the slope
       dipless  = y + gdips - a*x

       call return_linreg_dp(x,dipless,mask,a2,b)       

       a = a + a2

       if (abs((a-a_old)/a_old) < 1.d-4 .and. l > 1) exit

       ! With the new slope and offset estimates, make the flat data
       flaty = y - a*x - b

       gdips(:) = 0.d0

       do j = 1, ndips

          sigma        = 0.0
          mean         = 0.0
          amp          = 0.0

          ! Define first and last for indices - range to fit Gaussian to dip
          first = dips(j)
          last  = dips(j)
          if (dips(j) - fit_range < 1) then
             first = 1
          else if (dips(j) + fit_range > leng) then
             last = leng
          else 
             do i = dips(j) - fit_range, dips(j) + fit_range
                if (mask(i) == 0) cycle
                first = min(first,i)
                last  = max(last,i)
             end do
          end if

          ! allocate temporary array for voltage bins and rms bins
          allocate(x_tmp(last-first),y_tmp(last-first))
          allocate(count_tmp(last-first))
          allocate(mask_tmp(last-first))
          
          ! Flip the dip!
          do i = dips(j)-fit_range, dips(j)+fit_range
             ! skip over incides that are masked or outside the index range
             if (i < 1) cycle
             if (i > leng) cycle
             if (mask(i) == 0) cycle
             newy(i)  = -1.0*flaty(i)
          end do
          
          x_tmp(:)     = x(first:last)
          y_tmp(:)     = newy(first:last)
          count_tmp(:) = bincount(first:last) 
          mask_tmp(:)  = mask(first:last)
          
          write(dip_str,'(i0.2)') j
          
          ! Define estimates to the Gaussian parameters
          do i = dips(j), dips(j)+fit_range
             if (mask(i) == 0) cycle
             if (newy(i) < newy(dips(j))/2.0) then
                fwhm = x(i)-x(dips(j))
                exit
             end if
          end do
          
          sigma   = max(2.0*(fwhm/2.355), 0.00001)
          mean    = x(dips(j))
          amp     = maxval(y_tmp)
          
          pars(1) = mean
          pars(2) = sigma
          pars(3) = amp
          
          par_est = pars
          
          ! With our estimates, let's find the maximum likelihood values for the parameters
          do k = 1, ngibbs
             write(gibbstr,'(i0.3)') k
             do i = 1, 3
                currpar  = i
                ! define parameter uniform prior ranges
                if (i == 1) then
                   ! Ensure the mean value is within the dip range
                   P_uni(1) = par_est(1) - 2.0*par_est(2)
                   P_uni(2) = par_est(1) + 2.0*par_est(2)
                else if (i == 2) then
                   P_uni(1) = 0.75*par_est(2)
                   P_uni(2) = 1.25*par_est(2)
                else if (i == 3) then
                   ! Ensure amplitude is always greater than 0
                   P_uni(1) = 0.0
                   P_uni(2) = 1.5*par_est(3)
                end if
                
                pars(i) = maxlike_gauss_par(i)
             end do
          end do

          deallocate(x_tmp,y_tmp)
          deallocate(count_tmp)
          deallocate(mask_tmp)
          
          if (amp < 0.0) cycle
          if (sigma < 0.0) cycle
          
          gmodel = return_gaussian_dp(x,pars)
          gdips  = gdips + gmodel          
       end do

       model = a*x - gdips !+ b

    end do

  contains

    ! Grid out and solve for maximum likelihood parameter value
    function maxlike_gauss_par(par_i) result(gpar)
      use healpix_types
      implicit none
      ! real(sp),   intent(inout) :: gpar
      integer(i4b), intent(in)  :: par_i
      real(dp), dimension(1000) :: lnL, grid
      real(dp)                  :: gpar, tmp
      integer(i4b)              :: l, i, ind
      real(dp), dimension(last-first) :: gauss

      lnL(:) = 0.0

      do l = 1, 1000
         grid(l) = (P_uni(2)-P_uni(1))*(l-1)/1000 + P_uni(1)
         
         tmp = pars(par_i)
         pars(par_i) = grid(l)

         gauss = return_gaussian_dp(x_tmp,pars)
         do i = 1, last-first
            if (mask_tmp(i) == 0) cycle
            lnL(l) = lnL(l) - 0.50*(y_tmp(i) - gauss(i))**2/count_tmp(i)
         end do
         
      end do
      
      ind = maxloc(lnL,dim=1)

      gpar = grid(ind)

    end function maxlike_gauss_par
    
  end function return_gauss_lin_model_dp
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!! SP FUNCTIONS !!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  module subroutine return_linreg_sp(x,y,mask,slope,offset,trim)
    !=========================================================================
    ! Very simple function that fits the slope and offset for an x-y pair
    !
    ! Inputs:
    !
    ! x      : float array (sp)
    !          The independent variable
    ! y      : float array (sp)
    !          The dependent variable
    ! mask   : integer array
    !          which array values do not contribute to the fit? 
    ! trim   : logical (optional)
    !          are large deviations disregarded in the linear fit?
    !
    ! Output:
    !
    ! slope  : float (sp)
    !          The slope of the linear fit
    ! offset : float (sp)
    !          The offset of the linear fit
    !=========================================================================    
    implicit none
    
    real(sp),     dimension(:),  intent(in)    :: x, y
    integer(i4b), dimension(:),  intent(in)    :: mask
    logical(lgt), optional,      intent(in)    :: trim
    real(sp),                    intent(inout) :: slope, offset
    real(sp), dimension(:,:),    allocatable   :: adata, bdata
    real(sp), dimension(:),      allocatable   :: work
    real(sp), dimension(:),      allocatable   :: x2, y2
    integer(i4b)                               :: i, leng, count
    real(sp)                                   :: y_mean, y_var, y_std
    integer(i4b)                               :: info
    
    leng = size(x)

    y_mean = 0.0
    y_var  = 0.0
    y_std  = 0.0

    count = 0
    ! Count valid entries and start computing the mean
    do i = 1, leng
       if (mask(i) == 0) cycle
       count = count + 1
       y_mean = y_mean + y(i)
    end do
    ! If no entries, get the hell outta here
    if (count == 0) then
       slope = 0.d0
       offset = 0.d0
       return
    end if
    ! Now mean
    y_mean = y_mean/count
    
    ! Use the mean to compute the variance
    do i = 1, leng
       if (mask(i) == 0) cycle
       y_var  = y_var + (y(i)-y_mean)**2
    end do
    
    ! Normalize
    y_var = y_var/(count-1)
    
    ! Now standard deviations
    y_std  = sqrt(y_var)

    ! Prepare to build arrays for computing the regression coeffs using LAPACK
    allocate(x2(leng),y2(leng))

    x2(:) = 0.0
    y2(:) = 0.0

    ! Mask out bins that have no entries
    count = 0
    do i = 1, leng
       ! Mask out outliers (if (y < mean-std .or. y > mean + std))
       if (present(trim)) then
          if (y(i) < y_mean-2.0*y_std) cycle
          if (y(i) > y_mean+2.0*y_std) cycle
       end if
       if (mask(i) == 0) cycle
       count     = count + 1
       ! Dump the appropriate values to the arrays
       x2(count) = x(i)
       y2(count) = y(i) 
    end do

    allocate(adata(count,2), bdata(count,1), work(2*count))
    adata(:,:) = 0.0
    bdata(:,:) = 0.0
    work(:)    = 0.0
    
    adata(:,1) = 1.0
    adata(:,2) = x2(1:count)
    bdata(:,1) = y2(1:count)
    
    ! Let LAPACK do the work
    call sgels('N', count, 2, 1, adata, count, bdata, count, work, 2*count, info)
    
    slope  = bdata(2,1)
    offset = bdata(1,1)
    
    deallocate(adata)
    deallocate(bdata)
    deallocate(work)
    
  end subroutine return_linreg_sp

  subroutine dump_adc_in_out(self,name)
    implicit none

    class(comm_adc),   intent(inout) :: self
    character(len=50), intent(in)    :: name

    integer(i4b)                     :: nb, i

    nb = size(self%adc_in)

    if (self%myid == 0) then
       open(21, file=trim(self%outdir)//'/adc_in_'//trim(name)//'_dumped.dat') 
       open(22, file=trim(self%outdir)//'/adc_out_'//trim(name)//'_dumped.dat') 
       do i = 1, nb
          write(21,*) self%adc_in(i)
          write(22,*) self%adc_out(i)
       end do
       close(21)
       close(22)
    end if

  end subroutine dump_adc_in_out

end submodule comm_tod_adc_smod
