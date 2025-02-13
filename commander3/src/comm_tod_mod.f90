!===============================================================================
!
! Copyright (C) 2020 Institute of Theoretical Astrophysics, University of Oslo.
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
!================================================================================
module comm_tod_mod
  use comm_fft_mod
  use comm_huffman_mod
  use comm_conviqt_mod
  use comm_tod_orbdipole_mod
  use comm_tod_noise_psd_mod
  use comm_shared_arr_mod
  USE ISO_C_BINDING
  implicit none

  ! Structure for individual detectors
  type :: comm_detscan
     character(len=30) :: label                             ! Detector label
     real(dp)          :: gain, dgain, gain_invsigma        ! Gain; assumed constant over scan
     real(dp)          :: gain_def                          ! Default parameters
     real(dp)          :: chisq
     real(dp)          :: chisq_prop
     real(dp)          :: chisq_masked
     logical(lgt)      :: accept
     class(comm_noise_psd), pointer :: N_psd                            ! Noise PSD object
     real(sp),           allocatable, dimension(:)     :: tod            ! Detector values in time domain, (ntod)
     byte,               allocatable, dimension(:)     :: ztod           ! compressed values in time domain, (ntod)
     real(sp),           allocatable, dimension(:,:)   :: diode          ! (ndiode, ntod) array of undifferenced data
     type(byte_pointer), allocatable, dimension(:)     :: zdiode         ! pointers to the compressed undifferenced diode data, len (ndiode)
     byte,               allocatable, dimension(:)     :: flag           ! Compressed detector flag; 0 is accepted, /= 0 is rejected
     integer(i4b),       allocatable, dimension(:,:)   :: mask_dyn       ! Dynamic online-generated mask, (2,ntod), each row gives a range of masked samples
     type(byte_pointer), allocatable, dimension(:)     :: pix            ! pointer array of pixels length nhorn
     type(byte_pointer), allocatable, dimension(:)     :: psi            ! pointer array of psi, length nhorn
     integer(i4b),       allocatable, dimension(:,:)   :: offset_range   ! Beginning and end tod index of every offset region
     real(sp),           allocatable, dimension(:)     :: offset_level   ! Amplitude of every offset region(step)
     integer(i4b),       allocatable, dimension(:,:)   :: jumpflag_range ! Beginning and end tod index of regions where jumps occur
     real(dp),           allocatable, dimension(:)     :: baseline       ! Polynomial coefficients for baseline function
     integer(i4b),       allocatable, dimension(:,:)   :: pix_sol        ! Discretized pointing in solar centric coordinates, for zodi and sidelobe mapping
     integer(i4b),       allocatable, dimension(:,:)   :: psi_sol        ! Discretized polarization angle in solar centric coordinates, for zodi and sidelobe mapping

     ! Zodi sampling structures (downsampled and precomputed quantities. only allocated if zodi sampling is true)
     logical(lgt),       allocatable, dimension(:)    :: zodi_glitch_mask
     integer(i4b),       allocatable, dimension(:)    :: downsamp_pix
     real(sp),           allocatable, dimension(:)    :: downsamp_tod
     real(sp),           allocatable, dimension(:)    :: downsamp_sky
     real(sp),           allocatable, dimension(:)    :: downsamp_zodi
     real(sp),           allocatable, dimension(:, :) :: downsamp_scat
     real(sp),           allocatable, dimension(:, :) :: downsamp_therm
     real(sp),           allocatable, dimension(:, :) :: downsamp_point  ! (ntod,{lat_gal, lon_gal, lat_ecl, lon_ecl, solar elongation}

     real(sp),           allocatable, dimension(:, :) :: s_scat_lowres
     real(sp),           allocatable, dimension(:, :) :: s_therm_lowres
  end type comm_detscan

  ! Stores information about all detectors at once 
  type :: comm_scan
     integer(i4b)   :: ntod                                        ! Number of time samples
     integer(i4b)   :: ext_lowres(2)                            ! Shape of downgraded TOD including padding
     real(dp)       :: proctime    = 0.d0                          ! Processing time in seconds
     real(dp)       :: n_proctime  = 0                             ! Number of completed loops
     real(dp)       :: v_sun(3)                                    ! Observatory velocity relative to Sun in km/s
     real(dp)       :: t0(3)                                       ! MJD, OBT, SCET for start of chunk
     real(dp)       :: t1(3)                                       ! MJD, OBT, SCET for end f chunk
     real(dp)       :: x0_obs(3)                                   ! Observatory position (x,y,z) for start of chunk
     real(dp)       :: x1_obs(3)                                   ! Observatory position (x,y,z) for end f chunk
     real(dp)       :: x0_earth(3)                                 ! Observatory position (x,y,z) for start of chunk
     real(dp)       :: x1_earth(3)                                 ! Observatory position (x,y,z) for end f chunk

     type(huffcode) :: hkey                                        ! Huffman decompression key
     type(huffcode) :: todkey                                      ! Huffman decompression key
     integer(i4b)   :: chunk_num                                   ! Absolute number of chunk in the data files
     integer(i4b),        allocatable, dimension(:,:)   :: zext    ! Extension of compressed diode arrays
     real(sp),            allocatable, dimension(:)     :: downsamp_obs_time ! downsampled_obs_time used for zodi sampling
     class(comm_detscan), allocatable, dimension(:)     :: d       ! Array of all detectors
  end type comm_scan


  type, abstract :: comm_tod
     character(len=512) :: freq
     character(len=512) :: instlabel
     character(len=512) :: filelist
     character(len=512) :: procmaskf1
     character(len=512) :: procmaskf2
     character(len=512) :: procmaskfzodi
     character(len=512) :: initfile
     character(len=512) :: instfile
     character(len=512) :: operation
     character(len=512) :: outdir
     character(len=512) :: sims_output_dir  !< simulation folder
     character(len=512) :: noise_psd_model  
     character(len=512) :: level !which level of tod we want, L1 or L2
     logical(lgt) :: enable_tod_simulations !< simulation parameter to run commander3 in different regime
     logical(lgt) :: first_call
     logical(lgt) :: sample_L1_par                                ! If false, reduce L1 (diode) to L2 (detector) in precomputations
     logical(lgt) :: L2_exist
     character(len=512) :: L2file
     integer(i4b) :: comm, myid, numprocs                         ! MPI parameters
     integer(i4b) :: comm_shared, myid_shared, numprocs_shared    ! MPI parameters
     integer(i4b) :: comm_inter, myid_inter                       ! MPI parameters
     integer(i4b) :: band                                        ! Absolute band ID
     integer(i4b) :: id                                          ! Relative band ID
     integer(i4b) :: zodiband                                        ! Band ID for zodi
     integer(i4b) :: nmaps                                        ! Number of Stokes parameters
     integer(i4b) :: ndet                                         ! Number of active detectors
     integer(i4b) :: nhorn                                        ! Number of horns
     integer(i4b) :: ndiode                                      ! Number of diodes that makeup each detector
     character(len=10), allocatable, dimension(:,:)  :: diode_names  ! Names of each diode, (ndet, ndiode)
     integer(i4b) :: nscan, nscan_tot                             ! Number of scans
     integer(i4b) :: first_scan, last_scan
     integer(i4b) :: npsi                                         ! Number of discretized psi steps
     integer(i4b) :: flag0
     integer(i4b) :: n_xi                                         ! Number of noise parameters
     integer(i4b) :: ntime                                        ! Number of time values
     integer(i4b) :: baseline_order                               ! Polynomial order for baseline

     real(dp)     :: central_freq                                 !Central frequency
     real(dp)     :: samprate, samprate_lowres                    ! Sample rate in Hz
     real(dp)     :: chisq_threshold                              ! Quality threshold in sigma
     character(len=512) :: abscal_comps            ! List of components to calibrate against
     logical(lgt) :: compressed_tod               
     logical(lgt) :: apply_inst_corr               
     logical(lgt) :: sample_abs_bp
     logical(lgt) :: symm_flags               
     logical(lgt) :: HFI_flag 
     class(comm_orbdipole), pointer :: orb_dp
     real(dp), allocatable, dimension(:)     :: gain0                                      ! Mean gain
     real(dp), allocatable, dimension(:)     :: polang                                      ! Detector polarization angle
     real(dp), allocatable, dimension(:,:)   :: polang_prior
        ! Detector polarization angle prior [ndet,mean/rms]
     real(dp), allocatable, dimension(:)     :: mbang                                       ! Main beams angle
     real(dp), allocatable, dimension(:)     :: mono                                        ! Monopole
     real(dp), allocatable, dimension(:)     :: fwhm, elip, psi_ell                         ! Beam parameter
     real(dp), allocatable, dimension(:)     :: nu_c                                        ! Center frequency
     real(dp), allocatable, dimension(:,:,:) :: prop_bp         ! proposal matrix, L(ndet,ndet,ndelta),  for bandpass sampler
     real(dp), allocatable, dimension(:)     :: prop_bp_mean    ! proposal matrix, sigma(ndelta), for mean
     real(sp), allocatable, dimension(:,:)   :: xi_n_P_uni      ! Uniform prior for noise PSD parameters
     real(sp), allocatable, dimension(:)     :: xi_n_P_rms      ! RMS for active noise PSD prior
     real(sp), allocatable, dimension(:,:)   :: xi_n_nu_fit     ! Frequency range used to fit noise PSD parameters
     integer(i4b)      :: nside, nside_param                    ! Nside for pixelized pointing
     integer(i4b)      :: nobs, nobs_lowres                     ! Number of observed pixels for this core
     integer(i4b)      :: n_bp_prop                       ! Number of consecutive bandpass proposals in each main iteration; should be 2 for MH
     integer(i4b) :: output_n_maps                                ! Output n_maps
     character(len=512) :: init_from_HDF                          ! Read from HDF file
     character(len=512) :: datadir
     integer(i4b) :: output_4D_map                                ! Output 4D maps
     integer(i4b) :: output_aux_maps                              ! Output auxiliary maps
     integer(i4b) :: halfring_split                               ! Type of halfring split 0=None, 1=HR1, 2=HR2
     logical(lgt) :: subtract_zodi                                ! Subtract zodical light (defined in the parameter file)
     logical(lgt) :: sample_zodi                                  ! Sample zodi model parameters (defined in the parameter file)
     logical(lgt) :: output_zodi_comps                            ! Output zodi components
     logical(lgt) :: use_solar_point                              ! Compute solar centric pointing, for zodi or sidelobe mapping
     real(sp)     :: sol_elong_range(2)                           ! Acceptable solar elongation range
     logical(lgt) :: correct_sl                                   ! Subtract sidelobes
     logical(lgt) :: correct_orb                                  ! Subtract CMB dipole
     logical(lgt) :: sample_mono                                  ! Subtract detector-specific monopoles
     logical(lgt) :: orb_4pi_beam                                 ! Perform 4pi beam convolution for orbital CMB dipole 
     integer(i4b),       allocatable, dimension(:)     :: stokes  ! List of Stokes parameters
     real(dp),           allocatable, dimension(:,:,:) :: w       ! Stokes weights per detector per horn, (nmaps,nhorn,ndet)
     real(sp),           allocatable, dimension(:)     :: sin2psi  ! Lookup table of sin(2psi)
     real(sp),           allocatable, dimension(:)     :: cos2psi  ! Lookup table of cos(2psi)
     real(sp),           allocatable, dimension(:)     :: psi      ! Lookup table of psi
     real(dp),           allocatable, dimension(:,:)   :: L_prop_mono  ! Proposal matrix for monopole sampling
     real(dp),           allocatable, dimension(:,:)   :: satpos   ! Satellite position for all scans
     real(dp),           allocatable, dimension(:)     :: mjds     ! MJDs for all scans(nscan_tot)
     real(dp),           allocatable, dimension(:,:)   :: v_sun    ! Sun velocities for all scans (3, nscan_tot)
     type(spline_type)                                 :: x_obs_spline(3), x_earth_spline(3) ! splines to compute observer and earth positions
     type(comm_scan),    allocatable, dimension(:)     :: scans    ! Array of all scans
     integer(i4b),       allocatable, dimension(:)     :: scanid   ! List of scan IDs
     integer(i4b),       allocatable, dimension(:)     :: nscanprproc   ! List of scan IDs
     integer(i4b),       allocatable, dimension(:)     :: partner ! Partner detector; for symmetrizing flags
     integer(i4b),       allocatable, dimension(:)     :: horn_id  ! Internal horn number per detector
     real(dp),           dimension(4)                  :: x_im    ! feedhorn imbalance parameters, with duplicates
     character(len=512), allocatable, dimension(:)     :: hdfname  ! List of HDF filenames for each ID
     character(len=512), allocatable, dimension(:)     :: label    ! Detector labels
     class(comm_map), pointer                          :: procmask => null() ! Mask for gain and n_corr
     class(comm_map), pointer                          :: procmask2 => null() ! Mask for gain and n_corr
     class(comm_map), pointer                          :: procmask_zodi => null() ! Mask for sampling zodi
     !class(comm_map), pointer                          :: mask_solar => null() ! Solar centric/sidelobe mask
     real(dp),           allocatable, dimension(:,:)   :: mask_solar           ! Solar centric/sidelobe mask
     logical(lgt)                                      :: map_solar_allocated
     real(dp),           pointer,     dimension(:,:)   :: map_solar           ! Full-sky solar centric/sidelobe model
!     class(comm_map), pointer                          :: map_solar => null() ! Solar centric/sidelobe model
     class(comm_mapinfo), pointer                      :: info => null()    ! Map definition
     class(comm_mapinfo), pointer                      :: slinfo => null()  ! Sidelobe map info
     class(comm_mapinfo), pointer                      :: mbinfo => null()  ! Main beam map info
     class(map_ptr),     allocatable, dimension(:)     :: slbeam, mbeam   ! Sidelobe beam data (ndet)
     class(conviqt_ptr), allocatable, dimension(:)     :: slconv ! SL-convolved maps (ndet)
     class(conviqt_ptr), allocatable, dimension(:)     :: slconvA, slconvB ! SL-convolved maps (ndet)
     real(dp),           allocatable, dimension(:,:)   :: bp_delta  ! Bandpass parameters (0:ndet, npar)
     real(dp),           allocatable, dimension(:,:)   :: spinaxis ! For load balancing
     integer(i4b),       allocatable, dimension(:)     :: pix2ind ! Array mapping all npix pixels to the uniquely observed pixels in the tod object for saving memory
     integer(i4b),       allocatable, dimension(:)     :: ind2pix, ind2sl ! Lookup tables used with pix2ind 
     real(sp),           allocatable, dimension(:,:)   :: ind2ang ! Lookup tables used with pix2ind for pixel angles
     real(dp),           allocatable, dimension(:,:)   :: ind2vec ! Lookup tables used with pix2ind for pixel unit vectors
     real(dp),           allocatable, dimension(:,:)   :: ind2vec_ecl ! Lookuptable for lowres ind to ecliptic unit vector
     real(dp),           allocatable, dimension(:,:)   :: ind2vec_ecl_lowres ! Lookuptable for lowres ind to ecliptic unit vector
     integer(i4b),       allocatable, dimension(:)     :: udgrade_pix_zodi !Lookuptable for highres pix to lowres pix
     integer(i4b),       allocatable, dimension(:)     :: pix2ind_lowres !Lookuptable for lowres zodi pixels

     character(len=128)                                :: tod_type
     integer(i4b)                                      :: nside_beam
     integer(i4b)                                      :: verbosity ! verbosity of output
     integer(i4b),       allocatable, dimension(:,:)   :: jumplist  ! List of stationary periods (ndet,njump+2)
     real(dp)                                          :: accept_threshold ! Required fraction of unflagged data in a detscan in order to be accepted 
     logical(lgt)                                      :: orbital ! flag for whether the orbital or solar dipole is used as the template in construct_dipole_template()
     ! Gain parameters
     logical(lgt)                            :: gain_tune_sigma0
     real(dp)                                :: gain_samprate
     real(dp), allocatable, dimension(:)     :: gain_sigma_0  ! size(ndet), the estimated white noise level of that scan. Not truly a white noise since our model is sigma_0**2 * (f/fknee)**alpha instead of sigma_0 ** 2 (1 + f/fknee ** alpha)
     real(dp), allocatable, dimension(:)    :: gain_fknee ! size(ndet)
     real(dp), allocatable, dimension(:)    :: gain_alpha ! size(ndet)
     real(dp) :: gain_sigma0_std ! std for metropolis-hastings sampling
     real(dp) :: gain_fknee_std ! std for metropolis-hastings sampling
     real(dp) :: gain_alpha_std ! std for metropolis-hastings sampling
     integer(i4b), allocatable, dimension(:) :: split

     ! Zodi parameters and spline objects
     integer(i4b) :: zodi_n_comps
   !   real(sp), allocatable, dimension(:, :, :) :: zodi_scat_cache, zodi_therm_cache ! Cached s_zodi array for a given processor
     real(sp), allocatable, dimension(:, :, :) :: zodi_scat_cache, zodi_therm_cache ! Cache for zodi
     real(sp), allocatable, dimension(:, :, :) :: zodi_scat_cache_lowres, zodi_therm_cache_lowres ! Cache for lowresolution zodi (used for sampling)
     real(dp)                                  :: zodi_cache_time, zodi_init_cache_time! Time of cached zodi array
     !real(dp), allocatable, dimension(:)       :: zodi_emissivity, zodi_albedo ! sampled parameters
     real(dp), allocatable, dimension(:, :)    :: zodi_spl_phase_coeffs
     real(dp), allocatable, dimension(:)       :: zodi_spl_solar_irradiance, zodi_phase_func_normalization
     type(spline_type), allocatable            :: zodi_b_nu_spl_obj(:)
     logical(lgt)                              :: zodi_tod_params_are_initialized, zodi_scattering, udgrade_zodi
   contains
     procedure                           :: read_tod
     procedure                           :: diode2tod_inst
     procedure                           :: read_tod_inst
     procedure                           :: read_scan_inst
     procedure                           :: get_scan_ids
     procedure                           :: dumpToHDF
     procedure                           :: dumpToHDF_inst
     procedure                           :: initHDF
     procedure                           :: initHDF_inst
     procedure                           :: get_det_id
     procedure                           :: initialize_bp_covar
     procedure(process_tod), deferred    :: process_tod
     procedure                           :: construct_sl_template
     procedure                           :: construct_corrtemp_inst
     procedure                           :: construct_dipole_template
     procedure                           :: construct_dipole_template_diff
     procedure                           :: output_scan_list
     procedure                           :: downsample_tod
     procedure                           :: compute_tod_chisq
     procedure                           :: get_total_chisq
     procedure                           :: symmetrize_flags
     procedure                           :: decompress_pointing_and_flags
     procedure                           :: decompress_tod
     procedure                           :: decompress_diodes
     procedure                           :: tod_constructor
     procedure                           :: load_instrument_file
     procedure                           :: load_instrument_inst
     procedure                           :: precompute_lookups
     procedure                           :: read_jumplist
     procedure                           :: remove_fixed_scans
     procedure                           :: apply_map_precond
     procedure                           :: collect_v_sun
     procedure                           :: precompute_zodi_lookups
     procedure                           :: clear_zodi_cache
     procedure                           :: create_dynamic_mask
     procedure                           :: get_s_static
  end type comm_tod

  abstract interface
     subroutine process_tod(self, chaindir, chain, iter, handle, map_in, delta, map_out, rms_out, map_gain)
       import i4b, comm_tod, comm_map, map_ptr, dp, planck_rng
       implicit none
       class(comm_tod),                     intent(inout) :: self
       character(len=*),                    intent(in)    :: chaindir
       integer(i4b),                        intent(in)    :: chain, iter
       type(planck_rng),                    intent(inout) :: handle
       type(map_ptr),     dimension(:,:),   intent(inout) :: map_in
       real(dp),          dimension(:,:,:), intent(inout) :: delta
       class(comm_map),                     intent(inout) :: map_out
       class(comm_map),                     intent(inout) :: rms_out
       type(map_ptr),     dimension(:,:),   intent(inout), optional :: map_gain
     end subroutine process_tod
  end interface

  type tod_pointer
    class(comm_tod), pointer :: p => null()
  end type tod_pointer

contains

  subroutine initialize_tod_mod(cpar)
    implicit none
    type(comm_params),       intent(in) :: cpar

    logical(lgt), save :: initialized = .false.

    if (initialized) return

    call initialize_fft_mod(cpar)
  end subroutine initialize_tod_mod

  subroutine tod_constructor(self, cpar, id, id_abs, info, tod_type)
    ! 
    ! Common constructor function for all TOD objects; allocatates and initializes general
    ! data structures. This routine is typically called from within an instrument-specific 
    ! initialization routine, *after* defining fields such that nhorn, ndet etc.
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_tod)
    !           TOD object to be initialized
    ! cpar:     derived type
    !           Object containing parameters from the parameterfile.
    ! id:       integer
    !           The index of the current band ampng accepted bands
    ! id_abs:   integer
    !           The index of the current band within the parameters, related to cpar

    ! info:     map_info structure
    !           Information about the maps for this band, like how the maps are distributed in memory
    ! tod_type: string
    !           Instrument specific tod type
    !
    ! Returns
    ! ----------
    ! None, but updates self
    !
    implicit none
    class(comm_tod),                intent(inout)  :: self
    integer(i4b),                   intent(in)     :: id, id_abs
    type(comm_params),              intent(in)     :: cpar
    class(comm_mapinfo),            target         :: info
    character(len=128),             intent(in)     :: tod_type

    integer(i4b) :: i, ndelta, ierr, unit
    character(len=512) :: datadir, solar_init

    self%id            = id
    self%band          = id_abs
    self%tod_type      = tod_type
    self%myid          = cpar%myid_chain
    self%comm          = cpar%comm_chain
    self%numprocs      = cpar%numprocs_chain
    self%myid_shared   = cpar%myid_shared
    self%comm_shared   = cpar%comm_shared
    self%myid_inter    = cpar%myid_inter
    self%comm_inter    = cpar%comm_inter
    self%info          => info
    self%init_from_HDF = cpar%ds_tod_initHDF(id_abs)
    self%freq          = cpar%ds_label(id_abs)
    self%instlabel     = cpar%ds_instlabel(id_abs)
    self%operation     = cpar%operation
    self%outdir        = cpar%outdir
    self%first_call    = .true.
    self%first_scan    = cpar%ds_tod_scanrange(id_abs,1)
    self%last_scan     = cpar%ds_tod_scanrange(id_abs,2)
    self%flag0         = cpar%ds_tod_flag(id_abs)
    self%abscal_comps  = cpar%ds_tod_abscal(id_abs)
    self%nscan_tot     = cpar%ds_tod_tot_numscan(id_abs)
    self%output_4D_map = cpar%output_4D_map_nth_iter
    self%output_aux_maps = cpar%output_aux_maps
    self%output_zodi_comps = cpar%zs_output_comps
    self%central_freq  = cpar%ds_nu_c(id_abs)
    self%halfring_split= cpar%ds_tod_halfring(id_abs)
    self%nside_param   = cpar%ds_nside(id_abs)
    self%verbosity     = cpar%verbosity
    self%sims_output_dir = cpar%sims_output_dir
    self%apply_inst_corr = .false.
    self%enable_tod_simulations = cpar%enable_tod_simulations
    self%accept_threshold = 0.9d0 ! default
    self%level        = cpar%ds_tod_level(id_abs)
    self%sample_abs_bp   = .false.
    self%zodiband        = -1
    self%sol_elong_range = [0., 180.]
    
    if (cpar%include_tod_zodi) then
      self%subtract_zodi = cpar%ds_tod_subtract_zodi(self%band)
      self%zodi_n_comps = cpar%zs_ncomps
      self%sample_zodi = cpar%sample_zodi .and. self%subtract_zodi
   end if
   self%use_solar_point = self%subtract_zodi

    if (trim(self%tod_type)=='SPIDER') then
      self%orbital = .false.
    else
      self%orbital = .true.
    end if

    if (trim(self%noise_psd_model) == 'oof') then
       self%n_xi = 3  ! {sigma0, fknee, alpha}
    else if (trim(self%noise_psd_model) == '2oof') then
       self%n_xi = 5  ! {sigma0, fknee, alpha, fknee2, alpha2}
    else if (trim(self%noise_psd_model) == 'oof_gauss') then
       self%n_xi = 6  ! {sigma0, fknee, alpha, amp, loc, sigma}
    else if (trim(self%noise_psd_model) == 'oof_quad') then
       self%n_xi = 5  ! {sigma0, fknee, alpha, slope, intercept}
    else
       write(*,*) 'Error: Invalid noise PSD model = ', trim(self%noise_psd_model)
       stop
    end if

    call mpi_comm_size(cpar%comm_shared, self%numprocs_shared, ierr)

    if (self%first_scan > self%last_scan) then
       write(*,*) 'Error: First scan larger than last scan for ', trim(self%freq)
       call mpi_finalize(ierr)
       stop
    end if

    datadir = trim(cpar%datadir)//'/'
    self%datadir     = datadir
    self%filelist    = trim(cpar%ds_tod_filelist(id_abs))
    self%procmaskf1  = trim(cpar%ds_tod_procmask1(id_abs))
    self%procmaskf2  = trim(cpar%ds_tod_procmask2(id_abs))
    self%instfile    = trim(cpar%ds_tod_instfile(id_abs))

    if (trim(self%level) == 'L1') then
        if (.not. self%sample_L1_par) then
          unit        = getlun()
          self%L2file = trim(self%datadir) // 'precomp_L2_'//trim(self%freq)//'.h5'
          inquire(file=trim(self%L2file), exist=self%L2_exist)
       else
          self%L2_exist = .false.
       end if
    else
       self%L2_exist = .true.
    end if

    call self%get_scan_ids(self%filelist)

    self%procmask => comm_map(self%info, self%procmaskf1)
    self%procmask2 => comm_map(self%info, self%procmaskf2)
    if (self%sample_zodi .and. self%subtract_zodi) then
      self%procmaskfzodi = trim(cpar%ds_tod_procmask_zodi(id_abs))
      self%procmask_zodi => comm_map(self%info, self%procmaskfzodi)
    end if
    if (trim(cpar%ds_tod_solar_mask(id_abs)) /= 'none') then
       !self%mask_solar => comm_map(self%info, cpar%ds_tod_solar_mask(id_abs))
       allocate(self%mask_solar(0:12*self%nside_param**2-1,1))
       call read_map(cpar%ds_tod_solar_mask(id_abs), self%mask_solar)
    end if
    
    do i = 0, self%info%np-1
       if (any(self%procmask%map(i,:) < 0.5d0)) then
          self%procmask%map(i,:) = 0.d0
       else
          self%procmask%map(i,:) = 1.d0
       end if
       if (any(self%procmask2%map(i,:) < 0.5d0)) then
          self%procmask2%map(i,:) = 0.d0
       else
          self%procmask2%map(i,:) = 1.d0
       end if
    end do

    self%nmaps    = info%nmaps
    !TODO: this should be changed to not require a really long string
    if (index(cpar%ds_tod_dets(id_abs), '.txt') /= 0) then
      self%ndet = count_detectors(cpar%ds_tod_dets(id_abs))
    else
      self%ndet = num_tokens(trim(cpar%ds_tod_dets(id_abs)), ",")
    end if

    ! Initialize jumplist
    call self%read_jumplist(cpar%ds_tod_jumplist(id_abs))

    allocate(self%stokes(self%nmaps))
    allocate(self%w(self%nmaps, self%nhorn, self%ndet))
    allocate(self%label(self%ndet))
    allocate(self%partner(self%ndet))
    allocate(self%horn_id(self%ndet))
    allocate(self%diode_names(self%ndet, self%ndiode))
    self%stokes = [1,2,3]
    self%w      = 1.d0
    self%x_im   = 0d0

    if (trim(cpar%ds_bpmodel(id_abs)) == 'additive_shift') then
       ndelta = 1
    else if (trim(cpar%ds_bpmodel(id_abs)) == 'powlaw_tilt') then
       ndelta = 1
    else
       write(*,*) 'Unknown bandpass model:', trim(cpar%ds_bpmodel(id_abs))
       stop
    end if
    allocate(self%bp_delta(0:self%ndet,ndelta))
    self%bp_delta = 0.d0

    !Allocate and initialize gain PSD Wiener filter structures; set to LFI defaults for now
    allocate(self%gain_sigma_0(self%ndet))
    allocate(self%gain_fknee(self%ndet))
    allocate(self%gain_alpha(self%ndet))
    self%gain_tune_sigma0 = .true.
    self%gain_samprate    = 1.d0 / 3600.d0
    self%gain_sigma_0     = 3d-4
    self%gain_fknee       = self%gain_samprate ! In seconds - this value is not necessarily set in stone and will be updated over the course of the run.
    self%gain_alpha       = -1.d0              ! This value is not necessarily set in stone and will be updated over the course of the run.
    self%gain_sigma0_std  = abs(self%gain_sigma_0(1) * 0.01)
    self%gain_fknee_std   = abs(self%gain_fknee(1) * 0.01)
    self%gain_alpha_std   = abs(self%gain_alpha(1) * 0.01)

    ! Allocate orbital dipole object; this should go in the experiment files, since it must be done after beam init
    !allocate(self%orb_dp)
    !self%orb_dp => comm_orbdipole(self%mbeam)
  end subroutine tod_constructor

  subroutine precompute_lookups(self)
    ! 
    ! Routine that precomputes static look-up tables in a given TOD object (pix2ind, ind2pix, ind2sl, ind2ang). 
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_tod)
    !           TOD object to be initialized
    ! Returns
    ! ----------
    ! None, but updates self
    !
    implicit none
    class(comm_tod),                intent(inout)  :: self

    real(dp)     :: f_fill, f_fill_lim(3), theta, phi
    integer(i4b) :: i, j, k, l, ierr
    integer(i4b), allocatable, dimension(:) :: pix

    ! Construct observed pixel array
    allocate(self%pix2ind(0:12*self%nside**2-1))
    self%pix2ind = -2
    do i = 1, self%nscan
       allocate(pix(self%scans(i)%ntod))
       if (self%nhorn == 2) then
          if (self%use_solar_point) allocate(self%scans(i)%d(1)%pix_sol(self%scans(i)%ntod,self%nhorn), self%scans(i)%d(1)%psi_sol(self%scans(i)%ntod,self%nhorn))
          do l = 1, self%nhorn
             call huffman_decode2_int(self%scans(i)%hkey, self%scans(i)%d(1)%pix(l)%p, pix)
             self%pix2ind(pix(1)) = -1
             do k = 2, self%scans(i)%ntod
                self%pix2ind(pix(k)) = -1
             end do
             if (self%use_solar_point) then
                call compute_solar_centered_pointing(self, i, j, pix, 0*pix, self%scans(i)%d(j)%pix_sol(:,l), self%scans(i)%d(j)%psi_sol(:,l))
             end if
          end do
       else
          do j = 1, self%ndet
             if (self%use_solar_point) allocate(self%scans(i)%d(j)%pix_sol(self%scans(i)%ntod,self%nhorn), self%scans(i)%d(j)%psi_sol(self%scans(i)%ntod,self%nhorn))
             do l = 1, self%nhorn
                call huffman_decode(self%scans(i)%hkey, self%scans(i)%d(j)%pix(l)%p, pix)
                self%pix2ind(pix(1)) = -1
                do k = 2, self%scans(i)%ntod
                   pix(k)  = pix(k-1)  + pix(k)
                   if (pix(k) > 12*self%nside**2-1) then
                       write(*,*) "Error: pixel number out of range for:"
                       write(*,*) "pixel nr", pix(k), "scan nr",  k, pix(1), l, "detector:", self%label(j), "chunk nr", self%scans(i)%chunk_num
                   end if
                   self%pix2ind(pix(k)) = -1
                end do
                if (self%use_solar_point) then
                   call compute_solar_centered_pointing(self, i, j, pix, 0*pix, self%scans(i)%d(j)%pix_sol(:,l), self%scans(i)%d(j)%psi_sol(:,l))
                end if
             end do
         end do
      end if
      deallocate(pix)
    end do
    self%nobs = count(self%pix2ind == -1)
    allocate(self%ind2pix(self%nobs))
    allocate(self%ind2sl(self%nobs))
    allocate(self%ind2ang(2,self%nobs))
    allocate(self%ind2vec(3,self%nobs))

    j = 1
    do i = 0, 12*self%nside**2-1
       if (self%pix2ind(i) == -1) then
          self%ind2pix(j) = i
          self%pix2ind(i) = j
          call pix2ang_ring(self%nside, i, theta, phi)
          call pix2vec_ring(self%nside, i, self%ind2vec(:,j))
          call ang2pix_ring(self%nside_beam, theta, phi, self%ind2sl(j))
          self%ind2ang(:,j) = [theta,phi]
          j = j+1
       end if
    end do

    f_fill = self%nobs/(12.*self%nside**2)
    call mpi_reduce(f_fill, f_fill_lim(1), 1, MPI_DOUBLE_PRECISION, MPI_MIN, 0, self%info%comm, ierr)
    call mpi_reduce(f_fill, f_fill_lim(2), 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, self%info%comm, ierr)
    call mpi_reduce(f_fill, f_fill_lim(3), 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, self%info%comm, ierr)
    if (self%myid == 0) then
       write(*,*) '|  Min/mean/max TOD-map f_sky = ', real(100*f_fill_lim(1),sp), real(100*f_fill_lim(3)/self%info%nprocs,sp), real(100*f_fill_lim(2),sp)
       write(*,*) '|'
    end if

  end subroutine precompute_lookups

  !**************************************************
  !             Utility routines
  !**************************************************

  subroutine load_instrument_file(self, nside_beam, nmaps_beam, pol_beam, comm_chain)
    implicit none
    class(comm_tod),   intent(inout) :: self
    integer(i4b),      intent(in)    :: nside_beam
    integer(i4b),      intent(in)    :: nmaps_beam
    logical(lgt),      intent(in)    :: pol_beam
    integer(i4b),      intent(in)    :: comm_chain 

    type(hdf_file) :: h5_file
    integer(i4b) :: lmax_beam, lmax_sl, i
    type(comm_mapinfo), pointer :: info_beam

    if(len(trim(self%instfile)) == 0) then
      write(*,*) "Cannot open instrument file with empty name for tod: " // self%tod_type
    end if
    allocate(self%fwhm(self%ndet))
    allocate(self%elip(self%ndet))
    allocate(self%psi_ell(self%ndet))
    allocate(self%nu_c(self%ndet))

    allocate(self%slbeam(self%ndet))
    allocate(self%mbeam(self%ndet))
    call open_hdf_file(self%instfile, h5_file, 'r')

    call read_hdf(h5_file, trim(adjustl(self%label(1)))//'/'//'sllmax', lmax_sl)
    call read_hdf(h5_file, trim(adjustl(self%label(1)))//'/'//'beamlmax', lmax_beam)
    self%slinfo => comm_mapinfo(comm_chain, nside_beam, lmax_sl, nmaps_beam, pol_beam)
    self%mbinfo => comm_mapinfo(comm_chain, nside_beam, lmax_beam, nmaps_beam, pol_beam)

    do i = 1, self%ndet
       call read_hdf(h5_file, trim(adjustl(self%label(i)))//'/'//'fwhm', self%fwhm(i))
       call read_hdf(h5_file, trim(adjustl(self%label(i)))//'/'//'elip', self%elip(i))
       call read_hdf(h5_file, trim(adjustl(self%label(i)))//'/'//'psi_ell', self%psi_ell(i))
       call read_hdf(h5_file, trim(adjustl(self%label(i)))//'/'//'centFreq', self%nu_c(i))
       !self%slbeam(i)%p => comm_map(self%slinfo, h5_file, .true., "sl", trim(self%label(i)))
       !self%mbeam(i)%p => comm_map(self%mbinfo, h5_file, .true., "beam", trim(self%label(i)))
       !call self%mbeam(i)%p%Y()
    end do

    call close_hdf_file(h5_file)

    self%nu_c   = self%nu_c * 1d9 ! Convert from Hz to GHz 
  end subroutine load_instrument_file


  subroutine read_tod(self, detlabels, datadir)
    ! 
    ! Reads common TOD fields into existing TOD object
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_tod)
    !           TOD object to be initialized
    ! detlabels: string (array)
    !           Array of detector labels, e.g., ["27M", "27S"]
    ! Returns
    ! ----------
    ! None, but updates self
    !
   implicit none
   class(comm_tod),                intent(inout)  :: self
   character(len=*), dimension(:), intent(in)     :: detlabels
   character(len=512), optional                   :: datadir
   
   
   integer(i4b) :: i, j, k, n, det, ierr, ndet_tot
   real(dp)     :: t1, t2
   real(sp)     :: psi
   type(hdf_file)     :: file
   character(len=128) :: buff_s
   
   integer(i4b), dimension(:), allocatable       :: ns
   real(dp), dimension(:), allocatable           :: mbang_buf, polang_buf
   character(len=100000)                         :: det_buf
   character(len=128), dimension(:), allocatable :: dets
   
    ! Read common fields
    allocate(self%polang(self%ndet))
    allocate(self%mbang(self%ndet))
    allocate(self%mono(self%ndet))
    allocate(self%gain0(0:self%ndet))
    self%mono = 0.d0
    if (self%myid == 0) then
       call open_hdf_file(self%initfile, file, "r")
       !TODO: figure out how to make this work
       call read_hdf_string2(file, "/common/det", det_buf, n)
       !call read_hdf(file, "/common/det",    det_buf)
       !write(det_buf, *) "27M, 27S, 28M, 28S"
       !write(det_buf, *) "18M, 18S, 19M, 19S, 20M, 20S, 21M, 21S, 22M, 22S, 23M, 23S"
       if (index(det_buf(1:n), '.txt') /= 0) then
         ndet_tot = count_detectors(det_buf(1:n))
       else
         ndet_tot = num_tokens(det_buf(1:n), ",")
       end if


       allocate(polang_buf(ndet_tot), mbang_buf(ndet_tot), dets(ndet_tot))
       polang_buf = 0
       mbang_buf = 0
       self%polang = 0
       self%mbang = 0
       if (index(det_buf(1:n), '.txt') /= 0) then
         call get_detectors(det_buf(1:n), dets)
       else
         call get_tokens(trim(adjustl(det_buf(1:n))), ',', dets)
       end if
      

!!$       do i = 1, ndet_tot
!!$          write(*,*) i, trim(adjustl(dets(i)))
!!$       end do
       !write(*,*) ndet_tot
       call read_hdf(file, "common/nside",  self%nside)
       if(self%nside /= self%nside_param) then
         write(*,*) "Nside=", self%nside_param, "found in parameter file does not match nside=", self%nside, "found in data files"
         stop
       end if
       if (self%nhorn == 2) then
         call read_hdf(file, "common/npsiA",   self%npsi)
       else
         call read_hdf(file, "common/npsi",   self%npsi)
       end if
       call read_hdf(file, "common/fsamp",  self%samprate)
       call read_hdf(file, "common/polang", polang_buf, opt=.true.)
       call read_hdf(file, "common/mbang",  mbang_buf, opt=.true.)
      !  do j = 1, ndet_tot
      !     print *,  j, trim(dets(j))
      !  end do
       do i = 1, self%ndet
          do j = 1, ndet_tot
             if(trim(adjustl(detlabels(i))) == trim(adjustl(dets(j)))) then
                exit
             end if
          end do
          if (j > ndet_tot) then
             write(*,*) ' Error -- detector not found in HDF file: ', trim(adjustl(detlabels(i)))
             stop
          end if
          self%polang(i) = 0.d0 !polang_buf(j)
          self%mbang(i)  = polang_buf(j)  !mbang_buf(j)
       end do
       deallocate(polang_buf, mbang_buf, dets)
       ! Read instrument specific parameters
       call self%read_tod_inst(file)

       call close_hdf_file(file)
    else
       call self%read_tod_inst
    end if
    call mpi_bcast(self%nside,    1,     MPI_INTEGER,          0, self%comm, ierr)
    call mpi_bcast(self%npsi,     1,     MPI_INTEGER,          0, self%comm, ierr)
    call mpi_bcast(self%samprate, 1,     MPI_DOUBLE_PRECISION, 0, self%comm, ierr)
    call mpi_bcast(self%polang,   self%ndet,  MPI_DOUBLE_PRECISION, 0, self%comm, ierr)
    call mpi_bcast(self%mbang,    self%ndet,  MPI_DOUBLE_PRECISION, 0, self%comm, ierr)

    call wall_time(t1)
    allocate(self%scans(self%nscan))
    call update_status(status, "000")
    do i = 1, self%nscan
       call read_hdf_scan(self%scans(i), self, self%hdfname(i), self%scanid(i), self%ndet, &
            & detlabels, self%nhorn, self%ndiode, self%diode_names)
    end do
    call update_status(status, "qqq")
!!$    if (self%ndiode > 1 .and. self%compressed_tod) then
!!$       ! Pre-allocate diode arrays, to avoid memory fragmenation
!!$       do i = 1, self%nscan
!!$          do j = 1, self%ndet
!!$             allocate(self%scans(i)%d(j)%zdiode(self%ndiode))
!!$             do k = 1, self%ndiode
!!$                allocate(self%scans(i)%d(j)%zdiode(k)%p(self%scans(i)%zext(j,k)))
!!$             end do
!!$          end do
!!$       end do
!!$    end if
    call update_status(status, "aaa")
    if (trim(self%level) == 'L2' .or. .not. self%L2_exist) then
       do i = 1, self%nscan
          call read_hdf_scan_data(self%scans(i), self, self%hdfname(i), self%scanid(i), self%ndet, &
               & detlabels, self%nhorn, self%ndiode, self%diode_names)

!!$       do j = 1, self%ndet
!!$          deallocate(self%scans(i)%d(j)%zdiode1,self%scans(i)%d(j)%zdiode2,self%scans(i)%d(j)%zdiode3,self%scans(i)%d(j)%zdiode4)
!!$       end do

          do det = 1, self%ndet
             if (allocated(self%scans(i)%d(det)%tod)) then
                self%scans(i)%d(det)%accept = all(self%scans(i)%d(det)%tod==self%scans(i)%d(det)%tod)
                if (.not. self%scans(i)%d(det)%accept) then
                   write(*,fmt='(a,i8,a,i3, i10)') 'Input TOD contain NaN -- scan =', &
                        & self%scanid(i), ', det =', det, count(self%scans(i)%d(det)%tod/=self%scans(i)%d(det)%tod)
                   write(*,fmt='(a,a)') '    filename = ', &
                        & trim(self%hdfname(i))
                end if
             end if
          end do
       end do
    end if
    call update_status(status, "111")
!!$    do i = self%nscan, 1, -1
!!$       do j = self%ndet, 1, -1
!!$          do k = 1, self%ndiode
!!$             deallocate(self%scans(i)%d(j)%zdiode(k)%p)
!!$          end do
!!$          deallocate(self%scans(i)%d(j)%zdiode)
!!$       end do
!!$    end do

    call update_status(status, "bbb")


    ! Initialize mean gain
    allocate(ns(0:self%ndet))
    self%gain0 = 0.d0
    ns         = 0
    do i = 1, self%nscan
       do j = 1, self%ndet
          if (.not. self%scans(i)%d(j)%accept) cycle
          self%gain0(j) = self%gain0(j) + self%scans(i)%d(j)%gain
          ns(j)         = ns(j) + 1
       end do
    end do
    call mpi_allreduce(MPI_IN_PLACE, self%gain0, self%ndet+1, &
         & MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)
    call mpi_allreduce(MPI_IN_PLACE, ns,         self%ndet+1, &
         & MPI_INTEGER,          MPI_SUM, self%comm, ierr)
    self%gain0(0) = sum(self%gain0)/sum(ns)
    where (ns > 0)
       self%gain0 = self%gain0 / ns - self%gain0(0)
    end where

    do i = 1, self%nscan
       do j = 1, self%ndet
          self%scans(i)%d(j)%dgain = self%scans(i)%d(j)%gain - self%gain0(0) - self%gain0(j)
       end do
    end do

    call update_status(status, "ccc")

    ! Precompute trigonometric functions
    allocate(self%sin2psi(self%npsi), self%cos2psi(self%npsi))
    allocate(self%psi(self%npsi))
    do i = 1, self%npsi
       psi             = (i-0.5)*2.0*pi/real(self%npsi,sp)
       self%psi(i)     = psi
       self%sin2psi(i) = sin(2.0*psi)
       self%cos2psi(i) = cos(2.0*psi)
    end do

    call mpi_barrier(self%comm, ierr)
    call wall_time(t2)
    if (self%myid == 0) write(*,fmt='(a,i4,a,i6,a,f8.1,a)') &
         & ' |  Myid = ', self%myid, ' -- nscan = ', self%nscan, &
         & ', TOD IO time = ', t2-t1, ' sec'


  end subroutine read_tod

  subroutine read_hdf_scan(self, tod, filename, scan, ndet, detlabels, nhorn, ndiode, diode_names)
    ! 
    ! Reads common scan information from TOD fileset
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_scan)
    !           Scan object
    ! tod:      derived class (comm_tod)
    !           Main TOD object to which current scan belongs
    ! filename: character
    !           TOD filename
    ! scan:     int
    !           Scan ID
    ! ndet:     int
    !           Number of detectors
    ! nhorn:    int
    !           Number of horns
    ! detlabels: string (array)
    !           Array of detector labels, e.g., ["27M", "27S"]
    ! ndiode:   int 
    !           Number of diodes per combined tod
    ! diode_names : string (array (ndet, ndiode)
    !           Array of diode labels, eg. [['sky00', 'sky01', 'load00',
    !           'load01'], ['sky10', 'sky11', 'load10', 'load11'], ...]
    ! scan:     derived class (comm_scan)
    !           
    !
    ! Returns
    ! ----------
    ! None, but updates scan object
    !
    ! TODO
    ! ----
    ! - ndet, nhorn and detlabels should be taken from tod, not inserted as separate parameters?
    ! 
    implicit none
    class(comm_scan),               intent(inout) :: self
    class(comm_tod),                intent(in)    :: tod
    character(len=*),               intent(in)    :: filename
    integer(i4b),                   intent(in)    :: scan, ndet, nhorn, ndiode
    character(len=*), dimension(:), intent(in)    :: detlabels
    character(len=*), dimension(:,:), intent(in)  :: diode_names

    integer(i4b)       :: i,j,k,l, n, m, ext(1)
    real(sp)           :: nu
    real(dp)           :: scalars(4)
    character(len=6)   :: slabel
    character(len=128) :: field
    type(hdf_file)     :: file
    integer(i4b), allocatable, dimension(:)       :: hsymb
    real(sp),     allocatable, dimension(:)       :: buffer_sp, xi_n, hsymb_sp
    integer(i4b), allocatable, dimension(:)       :: htree

    self%chunk_num = scan
    call int2string(scan, slabel)
    
    !print *, filename
    call open_hdf_file(filename, file, "r")

    ! Find array sizes
    call read_hdf(file, slabel // "/" // "common/ntod",   n)

    if (tod%halfring_split == 0) then
      m = get_closest_fft_magic_number(n)
    else if (tod%halfring_split == 1 .or. tod%halfring_split == 2) then
      m = get_closest_fft_magic_number(n/2)
    else 
      write(*,*) "Unknown halfring_split value in read_hdf_scan"
      stop
    end if
    if (real(m-n,dp)/real(n,dp) > 0.001d0) then
       write(*,*) 'Warning: More than 0.1% of scan', scan, ' removed by FFTW cut'
    end if

    self%ntod = m
    self%ext_lowres(1)   = -5    ! Lowres padding
    self%ext_lowres(2)   = int(self%ntod/int(tod%samprate/tod%samprate_lowres)) + 1 + self%ext_lowres(1)

    ! Read common scan data
    call read_hdf(file, slabel // "/common/vsun",  self%v_sun, opt=.true.)

    ! Read in time at the start and end of each scan (if available)
    call read_hdf(file, slabel // "/common/time",  self%t0)
    call read_hdf(file, slabel // "/common/time_end",  self%t1, opt=.true.)

    ! HKE: LFI files should be regenerated with (x,y,z) info
    ! Read in satellite and earth position at the start and end of each scan (if available)
    call read_hdf(file, slabel // "/common/satpos",  self%x0_obs, opt=.true.)
    call read_hdf(file, slabel // "/common/satpos_end",  self%x1_obs, opt=.true.)
    call read_hdf(file, slabel // "/common/earthpos",  self%x0_earth, opt=.true.)
    call read_hdf(file, slabel // "/common/earthpos_end",  self%x1_earth, opt=.true.)

    ! Read detector scans
    allocate(self%d(ndet), buffer_sp(n))
    if (tod%ndiode > 1 .and. tod%compressed_tod) allocate(self%zext(tod%ndet,tod%ndiode))
    do i = 1, ndet
       if ((i == 1 .and. nhorn == 2) .or. (nhorn .ne. 2)) then
         allocate(self%d(i)%psi(nhorn), self%d(i)%pix(nhorn))
       end if
       allocate(xi_n(tod%n_xi))

       field                = detlabels(i)
       self%d(i)%label      = trim(field)
       call read_hdf(file, slabel // "/" // trim(field) // "/scalars",   scalars)
      
       self%d(i)%gain_def   = scalars(1)
       self%d(i)%gain       = scalars(1)
       xi_n(1:3)            = scalars(2:4)
       xi_n(1)              = xi_n(1) * self%d(i)%gain_def ! Convert sigma0 to uncalibrated units
       self%d(i)%gain       = self%d(i)%gain_def
       self%d(i)%accept     = .true.

       if (tod%baseline_order >= 0) then
          allocate(self%d(i)%baseline(0:tod%baseline_order))
          self%d(i)%baseline = 0.
       end if

       if (trim(tod%noise_psd_model) == 'oof') then
         self%d(i)%N_psd => comm_noise_psd(xi_n, tod%xi_n_P_rms, tod%xi_n_P_uni, tod%xi_n_nu_fit)
       else if (trim(tod%noise_psd_model) == '2oof') then
          xi_n(4) =  1e-4  ! fknee2 (Hz); arbitrary value
          xi_n(5) = -1.000 ! alpha2; arbitrary value
          self%d(i)%N_psd => comm_noise_psd_2oof(xi_n, tod%xi_n_P_rms, tod%xi_n_P_uni, tod%xi_n_nu_fit)

       else if (trim(tod%noise_psd_model) == 'oof_gauss') then
          xi_n(4) =  0.00d0
          xi_n(5) =  1.35d0
          xi_n(6) =  0.40d0
          self%d(i)%N_psd => comm_noise_psd_oof_gauss(xi_n, tod%xi_n_P_rms, tod%xi_n_P_uni, tod%xi_n_nu_fit)

       else if (trim(tod%noise_psd_model) == 'oof_quad') then
          xi_n(4) =  0d0
          xi_n(5) =  0d0
          self%d(i)%N_psd => comm_noise_psd_oof_quad(xi_n, tod%xi_n_P_rms, tod%xi_n_P_uni, tod%xi_n_nu_fit)

!!$          open(58,file='noise.dat')
!!$          nu = 0.001d0 
!!$          do while (.true.)
!!$             write(58,*) nu, self%d(i)%N_psd%eval_full(nu)
!!$             nu = nu * 1.2d0
!!$             if (nu > tod%samprate) exit
!!$          end do
!!$          close(58)
!!$          stop

       end if
       deallocate(xi_n)

       ! Read Huffman coded data arrays
       if (nhorn == 2 .and. i == 1) then
         ! For a single DA, this is redundant, so we are loading 4 times the
         ! necessary pointing (and flags) information. Strictly speaking, this
         ! would involve needing to have a self%pixA and self%pixB attribute for
         ! WMAP only and not allocate self%d(i)%pix(j)
         do j = 1, nhorn 
           call read_hdf_opaque(file, slabel // "/" // trim(field) // "/pix" // achar(j+64),  self%d(i)%pix(j)%p)
           call read_hdf_opaque(file, slabel // "/" // trim(field) // "/psi" // achar(j+64),  self%d(i)%psi(j)%p)
         end do
       else if (nhorn .ne. 2) then
         do j = 1, nhorn
           call read_hdf_opaque(file, slabel // "/" // trim(field) // "/pix",  self%d(i)%pix(j)%p)
           call read_hdf_opaque(file, slabel // "/" // trim(field) // "/psi",  self%d(i)%psi(j)%p)
         end do
       end if
       call read_hdf_opaque(file, slabel // "/" // trim(field) // "/flag", self%d(i)%flag)

       ! Get compressed diode array sizes
!!$       if (tod%ndiode > 1 .and. tod%compressed_tod) then
!!$          call get_hdf_vlen_ext(file, slabel // '/' // trim(field) // '/diodes', self%zext(i,:))
!!$       end if

!!$       if(ndiode == 1) then
!!$         if (tod%compressed_tod) then
!!$            call read_hdf_opaque(file, slabel // "/" // trim(field) // "/tod", self%d(i)%ztod)
!!$         else
!!$            allocate(self%d(i)%tod(m))
!!$            call read_hdf(file, slabel // "/" // trim(field) // "/tod",    buffer_sp)
!!$            if (tod%halfring_split == 2 )then
!!$               self%d(i)%tod = buffer_sp(m+1:2*m)
!!$            else
!!$               self%d(i)%tod = buffer_sp(1:m)
!!$            end if
!!$         end if
!!$       else ! ndiode > 1 per tod
!!$          if(tod%compressed_tod == .false.) then
!!$             
!!$          else
!!$          end if
!!$          if (tod%compressed_tod) then
!!$             !allocate(self%d(i)%zdiode(ndiode))
!!$             !call read_hdf_vlen(file, slabel // '/' // trim(field) // '/diodes', self%d(i)%zdiode)
!!$             call read_hdf_vlen(file, slabel // '/' // trim(field) // '/diodes', self%d(i)%zdiode1, self%d(i)%zdiode2, self%d(i)%zdiode3, self%d(i)%zdiode4)
!!$             
!!$             !call read_hdf_opaque(file, slabel // '/' // trim(field) // '/' // trim(diode_names(i,k)), self%d(i)%zdiode(k)%p)
!!$          else
!!$             ! HKE: This array should have the ordering switched
!!$             allocate(self%d(i)%diode(ndiode, m))
!!$             do k = 1, ndiode
!!$                
!!$                call read_hdf(file, slabel // '/' // trim(field) // '/' //trim(diode_names(i, k)), buffer_sp)
!!$                if (tod%halfring_split == 2 )then
!!$                   self%d(i)%diode(k, :) = buffer_sp(m+1:2*m)
!!$                else
!!$                   self%d(i)%diode(k, :) = buffer_sp(1:m)
!!$                end if
!!$             end do
!!$          end if
!!$       end if

!!$       if (tod%compressed_tod) then
!!$          call read_hdf_opaque(file, slabel // "/" // trim(field) // "/ztod", self%d(i)%ztod)
!!$       else
!!$          allocate(self%d(i)%tod(m))
!!$          call read_hdf(file, slabel // "/" // trim(field) // "/tod",    buffer_sp)
!!$          if (tod%halfring_split == 2 )then
!!$             self%d(i)%tod = buffer_sp(m+1:2*m)
!!$          else
!!$             self%d(i)%tod = buffer_sp(1:m)
!!$          end if
!!$       end if
    end do
    deallocate(buffer_sp)


    ! Initialize Huffman key
    call read_alloc_hdf(file, slabel // "/common/huffsymb", hsymb)
    call read_alloc_hdf(file, slabel // "/common/hufftree", htree)
    call hufmak_precomp_int(hsymb,htree,self%hkey)
    deallocate(hsymb, htree)
    if (tod%compressed_tod) then
       call read_alloc_hdf(file, slabel // "/common/huffsymb2", hsymb_sp)
       call read_alloc_hdf(file, slabel // "/common/hufftree2", htree)
       call hufmak_precomp_sp(hsymb_sp,htree,self%todkey)
       deallocate(hsymb_sp, htree)
    end if

    ! Read instrument-specific infomation
    call tod%read_scan_inst(file, slabel, detlabels, self)

    ! Clean up
    call close_hdf_file(file)

  end subroutine read_hdf_scan

  subroutine read_hdf_scan_data(self, tod, filename, scan, ndet, detlabels, nhorn, ndiode, diode_names)
    ! 
    ! Reads common scan information from TOD fileset
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_scan)
    !           Scan object
    ! tod:      derived class (comm_tod)
    !           Main TOD object to which current scan belongs
    ! filename: character
    !           TOD filename
    ! scan:     int
    !           Scan ID
    ! ndet:     int
    !           Number of detectors
    ! nhorn:    int
    !           Number of horns
    ! detlabels: string (array)
    !           Array of detector labels, e.g., ["27M", "27S"]
    ! ndiode:   int 
    !           Number of diodes per combined tod
    ! diode_names : string (array (ndet, ndiode)
    !           Array of diode labels, eg. [['sky00', 'sky01', 'load00',
    !           'load01'], ['sky10', 'sky11', 'load10', 'load11'], ...]
    ! scan:     derived class (comm_scan)
    !           
    !
    ! Returns
    ! ----------
    ! None, but updates scan object
    !
    ! TODO
    ! ----
    ! - ndet, nhorn and detlabels should be taken from tod, not inserted as separate parameters?
    ! 
    implicit none
    class(comm_scan),               intent(inout) :: self
    class(comm_tod),                intent(in)    :: tod
    character(len=*),               intent(in)    :: filename
    integer(i4b),                   intent(in)    :: scan, ndet, nhorn, ndiode
    character(len=*), dimension(:), intent(in)    :: detlabels
    character(len=*), dimension(:,:), intent(in)  :: diode_names

    integer(i4b)       :: i,j,k,l, n, m, ext(1)
    real(sp)           :: nu
    real(dp)           :: scalars(4)
    character(len=6)   :: slabel
    character(len=128) :: field   
    type(hdf_file)     :: file
    real(sp),     allocatable, dimension(:)       :: buffer_sp

    call int2string(scan, slabel)
    call open_hdf_file(filename, file, "r")

    call read_hdf(file, slabel // "/" // "common/ntod",   n)

    if (tod%halfring_split == 0) then
      m = get_closest_fft_magic_number(n)
    else if (tod%halfring_split == 1 .or. tod%halfring_split == 2) then
      m = get_closest_fft_magic_number(n/2)
    else
      write(*,*) "Unknown halfring_split value in read_hdf_scan"
      stop
    end if

    ! Find array sizes
    ! Read detector scans
    if (.not. tod%compressed_tod) allocate(buffer_sp(n))
    do i = 1, ndet
       field = detlabels(i)
       if(ndiode == 1) then
         if (tod%compressed_tod) then
            call read_hdf_opaque(file, slabel // "/" // trim(field) // "/ztod", self%d(i)%ztod)
         else
            allocate(self%d(i)%tod(m))
            call read_hdf(file, slabel // "/" // trim(field) // "/tod",    buffer_sp)
            if (tod%halfring_split == 2 )then
               self%d(i)%tod = buffer_sp(m+1:2*m)
            else
               self%d(i)%tod = buffer_sp(1:m)
            end if
         end if
       else ! ndiode > 1 per tod
          if(tod%compressed_tod .eqv. .false.) then
             
          else
          end if
          if (tod%compressed_tod) then
             allocate(self%d(i)%zdiode(ndiode))
             call read_hdf_vlen(file, slabel // '/' // trim(field) // '/diodes', self%d(i)%zdiode)
             !call read_hdf_vlen(file, slabel // '/' // trim(field) // '/diodes', self%d(i)%zdiode1, self%d(i)%zdiode2, self%d(i)%zdiode3, self%d(i)%zdiode4)
             
             !call read_hdf_opaque(file, slabel // '/' // trim(field) // '/' // trim(diode_names(i,k)), self%d(i)%zdiode(k)%p)
          else
             ! HKE: This array should have the ordering switched
             allocate(self%d(i)%diode(ndiode, m))
             do k = 1, ndiode
                
                call read_hdf(file, slabel // '/' // trim(field) // '/' //trim(diode_names(i, k)), buffer_sp)
                if (tod%halfring_split == 2 )then
                   self%d(i)%diode(k, :) = buffer_sp(m+1:2*m)
                else
                   self%d(i)%diode(k, :) = buffer_sp(1:m)
                end if
             end do
          end if
       end if
    end do
    if (allocated(buffer_sp)) deallocate(buffer_sp)

    ! Clean up
    call close_hdf_file(file)

  end subroutine read_hdf_scan_data


  subroutine read_jumplist(self, jumplist)
    implicit none
    class(comm_tod),   intent(inout) :: self
    character(len=*),  intent(in)    :: jumplist

    integer(i4b) :: i, j, n_jumps, unit

    if (trim(jumplist) == 'none')  then

       allocate(self%jumplist(self%ndet, 2))
       self%jumplist(:,1) = 1
       self%jumplist(:,2) = 0 !self%nscan_tot

    else

       unit = getlun()
       open(unit,file=trim(jumplist))
       read(unit,*) n_jumps
       allocate(self%jumplist(self%ndet, n_jumps+2))
       self%jumplist(:,1) = 1
       do i = 1, n_jumps
          read(unit,*) j
          self%jumplist(:,i+1) = j
       end do
       self%jumplist(:,n_jumps+2) = 0  !self%nscan_tot
       close(unit)

    end if

  end subroutine read_jumplist


  subroutine get_scan_ids(self, filelist)
    implicit none
    class(comm_tod),   intent(inout) :: self
    character(len=*),  intent(in)    :: filelist

    integer(i4b)       :: unit, j, k, np, ind(1), i, n, m, n_tot, ierr, p, q, flen, c
    real(dp)           :: w_tot, w_curr, w, v0(3), v(3), spin(2)
    character(len=6)   :: fileid
    character(len=512) :: infile
    real(dp),           allocatable, dimension(:)   :: weight, sid
    real(dp),           allocatable, dimension(:,:) :: spinpos, spinaxis
    integer(i4b),       allocatable, dimension(:)   :: scanid, id, filenum
    integer(i4b),       allocatable, dimension(:)   :: proc
    real(dp),           allocatable, dimension(:)   :: pweight
    character(len=512), allocatable, dimension(:)   :: filename

    np = self%numprocs
    if (self%myid == 0) then
       unit = getlun()

       n_tot = 0
       open(unit, file=trim(filelist))
       read(unit,*) n
       do i = 1, n
          read(unit,*) j
          if (j >= self%first_scan .and. j <= self%last_scan) n_tot = n_tot+1
       end do
       close(unit)

       if (n_tot == 0) then
          write(*,*) 'Error: No accepted scans in filelist: ', trim(filelist)
          stop
       else if (n_tot==1) then
         self%nscan = n_tot
         open(unit, file=trim(filelist))
         read(unit,*) n
         allocate(filename(n_tot), scanid(n_tot), proc(n_tot), spinpos(2,n_tot), weight(n_tot))
         j = 1
         do i = 1, n
            !read(unit,*) scanid(j), filename(j), weight(j), spinpos(1:2,j)
            read(unit,*) p, infile, w, spin
            if (p < self%first_scan .or. p > self%last_scan) cycle
            scanid(j)      = p
            filename(j)    = infile
            weight(j)      = 2
            spinpos(1:2,j) = spin
         end do
         proc(1) = 0
         self%initfile = filename(1)
         close(unit)
         
       else
       
         open(unit, file=trim(filelist))
         read(unit,*) n
         allocate(id(n_tot), filename(n_tot), scanid(n_tot), weight(n_tot), proc(n_tot), pweight(0:np-1), sid(n_tot), spinaxis(n_tot,3), spinpos(2,n_tot), filenum(n_tot))
         j = 1
         filenum = 0
         do i = 1, n
            read(unit,*) p, infile, w, spin
            if (p < self%first_scan .or. p > self%last_scan) cycle
            scanid(j)      = p
            filename(j)    = infile
            weight(j)      = w
            spinpos(1:2,j) = spin
            id(j)          = j
            sid(j)         = scanid(j)
            if (self%enable_tod_simulations) then
               flen = len(trim(infile))
               read(infile(flen-8:flen-3),*) filenum(j)
            end if
            call ang2vec(spinpos(1,j), spinpos(2,j), spinaxis(j,1:3))
            if (j == 1) self%initfile = filename(j)
            j              = j+1
            if (j > n_tot) exit
         end do
         close(unit)

         if (self%enable_tod_simulations) then
            do i = 1, n_tot
               proc(i) = mod(filenum(i),self%numprocs)
            end do
         else
            ! Compute symmetry axis
            v0 = 0.d0
            do i = 2, n_tot
               v(1) = spinaxis(1,2)*spinaxis(i,3)-spinaxis(1,3)*spinaxis(i,2)
               v(2) = spinaxis(1,3)*spinaxis(i,1)-spinaxis(1,1)*spinaxis(i,3)
               v(3) = spinaxis(1,1)*spinaxis(i,2)-spinaxis(1,2)*spinaxis(i,1)
               if (v(3) < 0.d0) v  = -v
               if (sum(v*v) > 0.d0)  v0 = v0 + v / sqrt(sum(v*v))
            end do
            if (maxval(sqrt(v0*v0)) == 0) then
               v0 = 1
            else
               v0 = v0 / sqrt(v0*v0)
            end if
            
            do i = n_tot, 1, -1
               v(1) = spinaxis(1,2)*spinaxis(i,3)-spinaxis(1,3)*spinaxis(i,2)
               v(2) = spinaxis(1,3)*spinaxis(i,1)-spinaxis(1,1)*spinaxis(i,3)
               v(3) = spinaxis(1,1)*spinaxis(i,2)-spinaxis(1,2)*spinaxis(i,1)
               sid(i) = acos(max(min(sum(spinaxis(i,:)*spinaxis(1,:)),1.d0),-1.d0))
               if (sum(v*v0) < 0.d0) sid(i) = -sid(i) ! Flip sign 
            end do
            
            w_tot = sum(weight)
            if (self%enable_tod_simulations) then
               do i = 1, n_tot
                  infile = filename(i)
                  q = len(trim(infile))
                  read(infile(q-8:q-3),*) q
                  proc(i) = mod(q,np)
               end do
               pweight = 0.d0
               do k = 1, n_tot
                  pweight(proc(id(k))) = pweight(proc(id(k))) + weight(id(k))
               end do
            else if ((index(filelist, '-WMAP_') .ne. 0) .or. (index(filelist, '_DIRBE_') .ne. 0)) then
               pweight = 0d0
               ! Greedy after sorting
               ! Algorithm 2 of
               ! http://web.stanford.edu/class/msande319/Approximation%20Algorithm/lec1.pdf
               call QuickSort(id, weight)
               do i = n_tot, 1, -1
                 j = minloc(pweight, dim=1)
                 pweight(j-1) = pweight(j-1) + weight(i)
                 proc(id(i)) = j-1
               end do
            else 
               ! Sort by spin axis (Planck)
               proc    = -1
               call QuickSort(id, sid)
               w_curr = 0.d0
               j     = 1
               do i = np-1, 0, -1
                  w = 0.d0
                  do k = 1, n_tot
                     if (proc(k) == i) w = w + weight(k) 
                  end do
                  do while (w < w_tot/np .and. j <= n_tot)
                     proc(id(j)) = i
                     w           = w + weight(id(j))
                     if (w > 1.2d0*w_tot/np) then
                        ! Assign large scans to next core
                        proc(id(j)) = i-1
                        w           = w - weight(id(j))
                     end if
                     j           = j+1
                  end do
               end do
               do while (j <= n_tot)
                  proc(id(j)) = 0
                  j = j+1
               end do
               pweight = 0.d0
               do k = 1, n_tot
                  pweight(proc(id(k))) = pweight(proc(id(k))) + weight(id(k))
               end do
            end if

!!$            if (self%myid == 0) then
!!$               do k = 1, n_tot
!!$                  write(*,*) k, id(k), proc(k)
!!$               end do
!!$            end if
            
            write(*,*) '|'
            write(*,*) '|  Min/Max core weight = ', minval(pweight)/w_tot*np, maxval(pweight)/w_tot*np
            deallocate(id, pweight, weight, sid, spinaxis)
         end if

       ! Distribute according to consecutive PID
       do i = 1, n_tot
          proc(i) = max(min(int(real(i-1,sp)/real(n_tot-1,sp) * np),np-1),0)
       end do

         deallocate(filenum)

      end if
   end if

    call mpi_bcast(n_tot, 1,  MPI_INTEGER, 0, self%comm, ierr)
    if (self%myid /= 0) then
       allocate(filename(n_tot), scanid(n_tot), proc(n_tot), spinpos(2,n_tot))
    end if
    call mpi_bcast(filename, 512*n_tot,  MPI_CHARACTER, 0, self%comm, ierr)
    call mpi_bcast(scanid,       n_tot,  MPI_INTEGER,   0, self%comm, ierr)
    call mpi_bcast(proc,         n_tot,  MPI_INTEGER,   0, self%comm, ierr)
    call mpi_bcast(spinpos,    2*n_tot,  MPI_DOUBLE_PRECISION,   0, self%comm, ierr)

    self%nscan     = count(proc == self%myid)
    allocate(self%scanid(self%nscan), self%hdfname(self%nscan), self%spinaxis(self%nscan,2))
    j = 1
    do i = 1, n_tot
       if (proc(i) == self%myid) then
          self%scanid(j)     = scanid(i)
          self%hdfname(j)    = filename(i)
          self%spinaxis(j,:) = spinpos(:,i)
          j                  = j+1
       end if
    end do

    if (self%myid == 0) then
       allocate(self%nscanprproc(0:self%numprocs-1))
       do i = 0, self%numprocs-1
          self%nscanprproc(i) = count(proc == i)
       end do
    end if

    deallocate(filename, scanid, proc, spinpos)

  end subroutine get_scan_ids


  subroutine dumpToHDF(self, chainfile, iter, map, rms)
    implicit none
    class(comm_tod),                   intent(inout) :: self
    integer(i4b),                      intent(in)    :: iter
    type(hdf_file),                    intent(in)    :: chainfile
    class(comm_map),                   intent(in)    :: map, rms

    integer(i4b)       :: i, j, k, l, npar, ierr
    real(dp)           :: mu
    character(len=6)   :: itext
    character(len=512) :: path
    real(dp), allocatable, dimension(:,:,:) :: output
    real(dp), allocatable, dimension(:)     :: mjds

    npar = 3+self%n_xi
    if (self%baseline_order >= 0) npar = npar + self%baseline_order + 1
    allocate(output(self%nscan_tot,self%ndet,npar))
    allocate(  mjds(self%nscan_tot))

    ! Collect all parameters
    output = 0.d0
    mjds   = 0.d0
    do j = 1, self%ndet
       do i = 1, self%nscan
          k                         = self%scanid(i)
          output(k,j,1)             = self%scans(i)%d(j)%gain
          output(k,j,2)             = merge(1.d0,0.d0,self%scans(i)%d(j)%accept)
          output(k,j,3)             = self%scans(i)%d(j)%chisq
          output(k,j,4:3+self%n_xi) = self%scans(i)%d(j)%N_psd%xi_n
          if (self%baseline_order >= 0) then
             output(k,j,4+self%n_xi:npar) = self%scans(i)%d(j)%baseline
          end if
          if (j == 1) then
             mjds(k)                = self%scans(i)%t0(1)
          end if
       end do
    end do

    if (self%myid == 0) then
       call mpi_reduce(mpi_in_place, output, size(output), &
            & MPI_DOUBLE_PRECISION, MPI_SUM, 0, self%comm, ierr)
       call mpi_reduce(mpi_in_place, mjds, size(mjds), &
            & MPI_DOUBLE_PRECISION, MPI_SUM, 0, self%comm, ierr)
    else
       call mpi_reduce(output,       output, size(output), &
            & MPI_DOUBLE_PRECISION, MPI_SUM, 0, self%comm, ierr)
       call mpi_reduce(mjds,         mjds,   size(mjds), &
            & MPI_DOUBLE_PRECISION, MPI_SUM, 0, self%comm, ierr)
    end if

    if (self%myid == 0) then
       ! Fill in defaults (closest previous)
       do j = 1, self%ndet
          do i = 1, npar
             if (i >= 2 .and. i <= 4) cycle
             do k = 1, self%nscan_tot
                if (output(k,j,i) == 0.d0) then
                   l = k
                   if (k == 1) then
                      do while (output(l,j,i) == 0.d0 .and. l < self%nscan)
                         l = l+1
                      end do
                   else
                      do while (output(l,j,i) == 0.d0 .and. l > 1)
                         l = l-1
                      end do
                   end if
                   output(k,j,i) = output(l,j,i)
                end if
             end do
!!$             if (output(
!!$             mu = sum(output(:,j,i)) / count(output(:,j,i) /= 0.d0)
!!$             where (output(:,j,i) == 0.d0)
!!$                output(:,j,i) = mu
!!$             end where
          end do
       end do

!!$       do j = 1, self%ndet
!!$          do i = 1, 4
!!$             mu = sum(output(:,j,i)) / count(output(:,j,i) /= 0.d0)
!!$             where (output(:,j,i) == 0.d0)
!!$                output(:,j,i) = mu
!!$             end where
!!$          end do
!!$       end do

       call int2string(iter, itext)
       path = trim(adjustl(itext))//'/tod/'//trim(adjustl(self%freq))//'/'
       !write(*,*) 'path', trim(path)
       call create_hdf_group(chainfile, trim(adjustl(path)))
       call write_hdf(chainfile, trim(adjustl(path))//'gain',   output(:,:,1))
       call write_hdf(chainfile, trim(adjustl(path))//'accept', output(:,:,2))
       call write_hdf(chainfile, trim(adjustl(path))//'chisq',  output(:,:,3))
       call write_hdf(chainfile, trim(adjustl(path))//'xi_n',   output(:,:,4:3+self%n_xi))
       call write_hdf(chainfile, trim(adjustl(path))//'MJD',    mjds)
       if (self%baseline_order >= 0) call write_hdf(chainfile, trim(adjustl(path))//'baseline',   output(:,:,4+self%n_xi:npar))
       call write_hdf(chainfile, trim(adjustl(path))//'polang', self%polang)
       call write_hdf(chainfile, trim(adjustl(path))//'gain0',  self%gain0)
       call write_hdf(chainfile, trim(adjustl(path))//'x_im',   [self%x_im(1), self%x_im(3)])
       call write_hdf(chainfile, trim(adjustl(path))//'mono',   self%mono)
       call write_hdf(chainfile, trim(adjustl(path))//'bp_delta', self%bp_delta)
       call write_hdf(chainfile, trim(adjustl(path))//'gain_sigma_0', self%gain_sigma_0)
       call write_hdf(chainfile, trim(adjustl(path))//'gain_fknee', self%gain_fknee)
       call write_hdf(chainfile, trim(adjustl(path))//'gain_alpha', self%gain_alpha)
       
    end if

    call map%writeMapToHDF(chainfile, path, 'map')
    call rms%writeMapToHDF(chainfile, path, 'rms')

    ! Write instrument-specific parameters
    call self%dumpToHDF_inst(chainfile, path)

    deallocate(output, mjds)

  end subroutine dumpToHDF

  subroutine initHDF(self, chainfile, iter, map, rms)
    implicit none
    class(comm_tod),                   intent(inout) :: self
    integer(i4b),                      intent(in)    :: iter
    type(hdf_file),                    intent(in)    :: chainfile
    class(comm_map),                   intent(inout) :: map, rms

    integer(i4b)       :: i, j, k, npar, ierr, ext(3)
    character(len=6)   :: itext
    character(len=512) :: path
    real(dp), allocatable, dimension(:,:,:) :: output

    call int2string(iter, itext)
    path = trim(adjustl(itext))//'/tod/'//trim(adjustl(self%freq))//'/'

    call get_size_hdf(chainfile, trim(adjustl(path))//'xi_n', ext)
    npar = 2+ext(3)
    allocate(output(self%nscan_tot,self%ndet,npar))

    if (self%myid == 0) then
       call read_hdf(chainfile, trim(adjustl(path))//'gain',     output(:,:,1))
       call read_hdf(chainfile, trim(adjustl(path))//'accept',   output(:,:,2))
       call read_hdf(chainfile, trim(adjustl(path))//'xi_n',     output(:,:,3:2+self%n_xi))
       call read_hdf(chainfile, trim(adjustl(path))//'mono',     self%mono)
       call read_hdf(chainfile, trim(adjustl(path))//'bp_delta', self%bp_delta)
       call read_hdf(chainfile, trim(adjustl(path))//'gain0',    self%gain0)
       call read_hdf(chainfile, trim(adjustl(path))//'x_im', self%x_im(1:2))
       self%x_im(3) = self%x_im(2)
       self%x_im(4) = self%x_im(3)
       self%x_im(2) = self%x_im(1)
       call read_hdf(chainfile, trim(adjustl(path))//'gain_sigma_0',    self%gain_sigma_0)
       call read_hdf(chainfile, trim(adjustl(path))//'gain_fknee',    self%gain_fknee)
       call read_hdf(chainfile, trim(adjustl(path))//'gain_alpha',    self%gain_alpha)
    end if

    call mpi_bcast(output, size(output), MPI_DOUBLE_PRECISION, 0, &
         & self%comm, ierr)
    call mpi_bcast(self%bp_delta, size(self%bp_delta), MPI_DOUBLE_PRECISION, 0, &
         & self%comm, ierr)
!    call mpi_bcast(self%polang, size(self%polang), MPI_DOUBLE_PRECISION, 0, &
!         & self%comm, ierr)
    call mpi_bcast(self%mono, size(self%mono), MPI_DOUBLE_PRECISION, 0, &
         & self%comm, ierr)
    call mpi_bcast(self%gain0, size(self%gain0), MPI_DOUBLE_PRECISION, 0, &
         & self%comm, ierr)
!!$    call mpi_bcast(self%gain_sigma_0, size(self%gain_sigma_0), MPI_DOUBLE_PRECISION, 0, &
!!$         & self%comm, ierr)
!!$    call mpi_bcast(self%gain_fknee, size(self%gain_fknee), MPI_DOUBLE_PRECISION, 0, &
!!$         & self%comm, ierr)
!!$    call mpi_bcast(self%gain_alpha, size(self%gain_alpha), MPI_DOUBLE_PRECISION, 0, &
!!$         & self%comm, ierr)

!!$       self%gain_alpha = -2.5d0

!!$    do j = 1, self%ndet
!!$       where (output(:,j,1)>0.)
!!$          output(:,j,1) = 0.03d0 + j*0.01d0
!!$          !output(:,j,1) = output(:,j,1)- 0.02d0 !+ j*0.01d0
!!$       end where
!!$    end do

!!$    self%gain0(0) = sum(output(:,:,1))/count(output(:,:,1)>0.)
!!$    do j = 1, self%ndet
!!$       self%gain0(j) = sum(output(:,j,1))/count(output(:,j,1)>0.) - self%gain0(0)
!!$    end do

    do j = 1, self%ndet
       do i = 1, self%nscan
          k             = self%scanid(i)
          self%scans(i)%d(j)%gain                 = output(k,j,1)
          self%scans(i)%d(j)%dgain                = output(k,j,1)-self%gain0(0)-self%gain0(j)
          self%scans(i)%d(j)%N_psd%xi_n(1:ext(3)) = output(k,j,3:2+self%n_xi)
          !self%scans(i)%d(j)%N_psd%xi_n(1)        = self%scans(i)%d(j)%N_psd%xi_n(1) * 1d-2
          if (output(k,j,2) == 0) then
             self%scans(i)%d(j)%accept               = .false.  !output(k,j,5) == 1.d0
          end if
          !if (k > 20300                    .and. (trim(self%label(j)) == '26M' .or. trim(self%label(j)) == '26S')) self%scans(i)%d(j)%accept = .false.
          !if ((k > 24900 .and. k <= 25300) .and. (trim(self%label(j)) == '18M' .or. trim(self%label(j)) == '18S')) self%scans(i)%d(j)%accept = .false.
       end do
    end do

    call map%readMapFromHDF(chainfile, trim(adjustl(path))//'map')
    call rms%readMapFromHDF(chainfile, trim(adjustl(path))//'rms')

    ! Read instrument-specific parameters
    call self%initHDF_inst(chainfile, path)

    call self%remove_fixed_scans

    deallocate(output)

  end subroutine initHDF

  function get_det_id(self, label)
    implicit none
    class(comm_tod),                intent(inout) :: self
    character(len=*),               intent(in)    :: label
    integer(i4b)                                  :: get_det_id

    integer(i4b) :: i

    do i = 1, self%ndet
       !write(*,*) "From get_det_id:", trim(adjustl(label)), trim(adjustl(self%label(i)))
       !write(*,*) self%ndet
       if (trim(adjustl(label)) == trim(adjustl(self%label(i)))) then
          get_det_id = i
          return
       end if
    end do

    get_det_id = -1

  end function get_det_id

  subroutine initialize_bp_covar(self, filename)
    implicit none
    class(comm_tod),   intent(inout) :: self
    character(len=*),  intent(in)    :: filename

    integer(i4b) :: j, k, ndet, npar, unit, par, ios
    real(dp)     :: val
    character(len=25)   :: label, det1, det2
    character(len=1024) :: line

    unit = getlun()
    ndet = self%ndet
    npar = size(self%bp_delta,2)

    allocate(self%prop_bp(ndet,ndet,npar))
    allocate(self%prop_bp_mean(npar))
    self%prop_bp      = 0.d0
    self%prop_bp_mean = 0.d0


    open(unit,file=trim(filename), iostat=ios)
    if (ios .ne. 0) then
      write(*,*) 'Could not open ', trim(filename)
      stop
    end if
    do while (.true.)
       read(unit,'(a)',end=34) line
       line = trim(adjustl(line))
       if (line(1:1) == ' ' .or. line(1:1) == '#') then
          cycle
       else if (line(1:4) == 'INIT') then
          read(line,*) label, det1, par, val
          if (trim(adjustl(det1)) == 'MEAN') then
             self%bp_delta(0,par) = val
          else
             j = self%get_det_id(det1)
             if (j > 0) self%bp_delta(j,par) = val
          end if
       else if (line(1:4) == 'PROP') then
          read(line,*) label, det1, det2, par, val
          if (trim(adjustl(det1)) == 'MEAN') then
             self%prop_bp_mean(par) = sqrt(val)
          else
             j = self%get_det_id(det1)
             if (j < 0) cycle
             if (trim(adjustl(det2)) == trim(adjustl(det1))) then
                k = j
             else
                k = self%get_det_id(det2)
             end if
             if (k < 0) cycle
             
             self%prop_bp(j,k,par) = val
             self%prop_bp(k,j,par) = val
          end if
       else
          write(*,*) 'Unsupported entry in ', trim(filename)
          stop
       end if

    end do
34  close(unit)

!    if (maxval(abs(self%prop_bp)) == 0) then
!       write(*,*) 'Bandpass covariance file '//trim(filename)//' is improperly formatted'
!       stop
!    end if



    ! Compute square root; mean will be projected out after proposal generation
    do par = 1, npar
      call compute_hermitian_root(self%prop_bp(:,:,par), 0.5d0)
    end do

  end subroutine initialize_bp_covar


  !construct a sidelobe template in the time domain
  subroutine construct_sl_template(self, slconv, pix, psi, s_sl, polangle)
    implicit none
    class(comm_tod),                     intent(in)    :: self
    class(comm_conviqt),                 intent(in)    :: slconv
    integer(i4b),        dimension(:),   intent(in)    :: pix, psi
    real(dp),                            intent(in)    :: polangle
    real(sp),            dimension(:),   intent(out)   :: s_sl

    integer(i4b) :: j,k, pix_, bpsi, psii, psiu, subsamp
    real(dp)     :: psi_, unwrap, x0, x1

    real(dp), dimension(:), allocatable :: sub_sl, x_sl
    type(spline_type) :: spline

    subsamp = 5
    !subsamp = 1
    ! If I want to test, set subsamp = 1 and compare the sidelobe output

    allocate(sub_sl(size(pix)/subsamp), x_sl(size(pix)/subsamp))

    do j = 1, size(pix)/subsamp !TODO: determine a good subsampling factor. 10? 50?
       k = subsamp*(j-1) + 1
       pix_    = self%ind2sl(self%pix2ind(pix(k)))
       psi_    = self%psi(psi(k))-polangle

       sub_sl(j) = slconv%interp(pix_, psi_)
       x_sl(j) = k
    end do

    call spline_simple(spline, x_sl, sub_sl, regular=.true.)

    do j = 1, size(sub_sl)*subsamp  
      s_sl(j) = splint_simple(spline, real(j, dp))
    end do

    !do last few samples
    do j = size(sub_sl)*subsamp+1, size(s_sl)
      pix_    = self%ind2sl(self%pix2ind(pix(j)))
      psi_    = self%psi(psi(j))-polangle
      s_sl(j) = slconv%interp(pix_, psi_)
    end do

    deallocate(sub_sl, x_sl)
    call free_spline(spline)

  end subroutine construct_sl_template


!!$  !construct a sidelobe template in the time domain
!!$  subroutine construct_sl_template2(self, slconv, pix, psi, s_sl, polangle)
!!$    implicit none
!!$    class(comm_tod),                     intent(in)    :: self
!!$    class(comm_conviqt),                 intent(in)    :: slconv
!!$    integer(i4b),        dimension(:),   intent(in)    :: pix, psi
!!$    real(dp),                            intent(in)    :: polangle
!!$    real(sp),            dimension(:),   intent(out)   :: s_sl
!!$
!!$    integer(i4b) :: j, pix_, pix_prev, psi_prev
!!$    real(dp)     :: psi_
!!$
!!$    pix_prev = -1; psi_prev = -1
!!$    do j = 1, size(pix)
!!$       pix_    = self%ind2sl(self%pix2ind(pix(j)))
!!$       if (pix_prev == pix_ .and. psi(j) == psi_prev) then
!!$          s_sl(j) = s_sl(j-1)
!!$       else
!!$          psi_    = self%psi(psi(j))-polangle
!!$          s_sl(j) = slconv%interp(pix_, psi_)
!!$          pix_prev = pix_; psi_prev = psi(j)
!!$       end if
!!$    end do
!!$
!!$  end subroutine construct_sl_template2


  subroutine construct_corrtemp_inst(self, scan, pix, psi, s)
    !  Construct an instrument-specific correction template
    !
    !  Arguments:
    !  ----------
    !  self: comm_tod object
    !
    !  scan: int
    !       scan number
    !  pix: int
    !       index for pixel
    !  psi: int
    !       integer label for polarization angle
    !
    !  Returns:
    !  --------
    !  s:   real (sp)
    !       output template timestream
    implicit none
    class(comm_tod),                       intent(in)    :: self
    integer(i4b),                          intent(in)    :: scan
    integer(i4b),        dimension(:,:),   intent(in)    :: pix, psi
    real(sp),            dimension(:,:),   intent(out)   :: s

    s = 0.d0

  end subroutine construct_corrtemp_inst

  subroutine construct_dipole_template(self, scan, pix, psi, s_dip)
    !  construct a CMB dipole template in the time domain
    !
    !  Arguments:
    !  ----------
    !  self: comm_tod object
    !
    !  scan: int
    !       scan number
    !  pix: int
    !       index for pixel
    !  psi: int
    !       integer label for polarization angle
    !
    !  Returns:
    !  --------
    !  s_dip: real (sp)
    !       output dipole template timestream
    implicit none
    class(comm_tod),                   intent(inout) :: self
    integer(i4b),                      intent(in)    :: scan
    integer(i4b),    dimension(:,:),   intent(in)    :: pix, psi
    real(sp),        dimension(:,:),   intent(out)   :: s_dip

    integer(i4b) :: i, j, ntod
    real(dp)     :: v_ref(3)
    real(dp), allocatable, dimension(:,:) :: P
    logical(lgt)  :: relativistic

    relativistic = .true.

    ntod = self%scans(scan)%ntod

    allocate(P(3,ntod))
    do j = 1, self%ndet
       if (.not. self%scans(scan)%d(j)%accept) cycle
       if (self%orbital) then

         if (self%orb_4pi_beam) then
           v_ref = self%scans(scan)%v_sun
           do i = 1, ntod
               P(:,i) = [self%ind2ang(2,self%pix2ind(pix(i,j))), &
                       & self%ind2ang(1,self%pix2ind(pix(i,j))), &
                       & self%psi(psi(i,j))] ! [phi, theta, psi]
           end do
         else
            v_ref = self%scans(scan)%v_sun 
            do i = 1, ntod
             P(:,i) =  self%ind2vec(:,self%pix2ind(pix(i,j))) ! [v_x, v_y, v_z]
            end do
         end if

       else
          v_ref = v_solar
       end if
       call self%orb_dp%compute_CMB_dipole(j, v_ref, self%nu_c(j), &
            & self%orbital, self%orb_4pi_beam, P, s_dip(:,j))
    end do
    deallocate(P)

  end subroutine construct_dipole_template

  subroutine construct_dipole_template_diff(self, scan, pix, psi, orbital, horn_ind, s_dip, &
                                         &  factor)
    !construct a CMB dipole template in the time domain for differential data
    !
    !
    !  Arguments:
    !  ----------
    !  self: comm_tod object
    !
    !  scan: int
    !       scan number
    !  pix: int
    !       index for pixel
    !  psi: int
    !       integer label for polarization angle
    !  orbital: logical
    !       flag for whether the orbital or solar dipole is used as the template
    !  horn: integer
    !       corresponds to either horn = 1 (A) or horn = 2 (B)
    !  Returns:
    !  --------
    !  s_dip: real (sp)
    !       output dipole template timestream
    implicit none
    class(comm_tod),                   intent(inout) :: self
    integer(i4b),                      intent(in)    :: scan
    integer(i4b),    dimension(:,:),   intent(in)    :: pix, psi
    logical(lgt),                      intent(in)    :: orbital
    integer(i4b),                      intent(in)    :: horn_ind
    real(sp),        dimension(:,:),   intent(out)   :: s_dip
    real(dp),               intent(in), optional     :: factor

    integer(i4b) :: i, j, ntod
    real(dp)     :: v_ref(3), v_ref_next(3), f
    real(dp), allocatable, dimension(:,:) :: P
    logical(lgt)  :: relativistic

    relativistic = .true.

    f = 1.d0; if (present(factor)) f = factor
    ntod = self%scans(scan)%ntod

    allocate(P(3,ntod))
    j = 1
    if (self%orbital) then
       v_ref = self%scans(scan)%v_sun
       if (self%scanid(scan) == self%nscan_tot) then
          v_ref_next = v_ref
       else
          v_ref_next = self%v_sun(:,self%scanid(scan)+1)
       end if
    else
       v_ref      = v_solar
       v_ref_next = v_solar
    end if
    if (self%orb_4pi_beam) then
       do i = 1, ntod
          P(:,i) = [self%ind2ang(2,self%pix2ind(pix(i,j))), &
                  & self%ind2ang(1,self%pix2ind(pix(i,j))), &
                  & self%psi(psi(i,j))] ! [phi, theta, psi]
       end do
    else
       do i = 1, ntod
          P(:,i) =  self%ind2vec(:,self%pix2ind(pix(i,j))) ! [v_x, v_y, v_z]
       end do
    end if

    do j = 1, self%ndet
       ! Since the "det" index actually references which beam you're looking at,
       ! we need to reference horns A and B, which are ordered AABB in WMAP
       ! data.
       if (horn_ind == 1) then
          call self%orb_dp%compute_CMB_dipole(1, v_ref, self%nu_c(j), &
               & self%orbital, self%orb_4pi_beam, P, s_dip(:,j), factor=f, v_ref_next=v_ref_next)
       else if (horn_ind == 2) then
          call self%orb_dp%compute_CMB_dipole(3, v_ref, self%nu_c(j), &
               & self%orbital, self%orb_4pi_beam, P, s_dip(:,j), factor=f, v_ref_next=v_ref_next)
       else
          write(*,*) "Should only be 1 or 2"
          stop
       end if
    end do
    deallocate(P)

  end subroutine construct_dipole_template_diff

  subroutine output_scan_list(self, slist)
    implicit none
    class(comm_tod),                               intent(in)    :: self
    character(len=512), allocatable, dimension(:), intent(inout) :: slist

    integer(i4b)     :: i, j, mpistat(MPI_STATUS_SIZE), unit, ns, ierr, num_scan, n_buff
    character(len=4) :: pid


    call timer%start(TOD_WRITE)

    n_buff = 0
    do i = 1, self%nscan
       if (trim(slist(i)) == '') cycle
       n_buff = n_buff + 1
    end do
    call mpi_reduce(n_buff, num_scan,   1,  MPI_INTEGER,  MPI_SUM,  0, self%comm, ierr)
    call mpi_bcast(num_scan,   1,     MPI_INTEGER,    0, self%comm, ierr)
    if (self%myid == 0) then
       call int2string(self%myid, pid)
       unit = getlun()
       open(unit,file=trim(self%outdir)//'/filelist_'//trim(self%freq)//'.txt', recl=512)
       write(unit,*) num_scan
       do i = 1, self%nscan
          if (trim(slist(i)) == '') cycle
          write(unit,*) trim(slist(i))
       end do
       deallocate(slist)
       do j = 1, self%numprocs-1
          ns = self%nscanprproc(j)
          allocate(slist(ns))
          call mpi_recv(slist, 512*ns, MPI_CHARACTER, j, 98, self%comm, mpistat, ierr)
          do i = 1, ns
             if (trim(slist(i)) == '') cycle
             write(unit,*) trim(slist(i))
          end do
          deallocate(slist)
       end do
       close(unit)
    else
       call mpi_send(slist, 512*self%nscan, MPI_CHARACTER, 0, 98, self%comm, ierr)
       deallocate(slist)
    end if
    call timer%stop(TOD_WRITE)
  end subroutine output_scan_list



   subroutine downsample_tod(self, tod_in, ext, tod_out, mask, threshold, width)
      ! Downsamples a time-ordered signal by a moving average filter.
      !
      ! This function is used by calling it twice. In the first call, we providing it with only the 
      ! `tod_in` and `ext` arguments:  `call tod%downsample_tod(tod_in, ext)`. This populates the `ext` 
      ! list with the upper and lower bounds of the downsampled TOD. We then allocate the downsampled 
      ! `tod_out` as `allocate(downsampled_array(ext(1):ext(2)))`. Finally, we call the function again
      ! to get the downsampled tods: `tod%downsample_tod(tod_in, ext, downsampled_array)`.
      ! 
      ! Remember to set `tod_out` to zero before passing it to this function to avoid floating invalid.
      !
      ! Parameters
      ! ----------
      ! tod_in:
      !     The input TOD to be downsampled.
      ! ext:
      !     An integer array of length 2. The first element is the lower bound of the downsampled TOD,
      !     and the second element is the upper bound of the downsampled TOD.
      ! tod_out: optional
      !     The downsampled TOD. If not provided, the function will only populate the `ext` array.
      ! mask: optional
      !     A mask to apply to the TOD before downsampling. If not provided, no mask is applied.
      ! threshold: optional
      !     Sets all values in the downsampled tod to under the threshold to zero.
      ! step: optional
      !     The step size of the downsampled TOD. If not provided, the step size is determined by the
      !     ratio between the input TOD samplerate and the instrument lower resolution samplerate.
      !
      implicit none
      class(comm_tod),                    intent(in)     :: self
      real(sp), dimension(:),             intent(in)     :: tod_in
      integer(i4b),                       intent(inout)  :: ext(2)
      real(sp), dimension(ext(1):ext(2)), intent(out), optional :: tod_out
      real(sp), dimension(:),             intent(in),  optional :: mask
      real(sp),                           intent(in),  optional :: threshold
      real(dp),                           intent(in),  optional :: width

      integer(i4b) :: i, j, k, m, n, ntod, w, npad

      ntod = size(tod_in)
      npad = 5
      if (present(width)) then
         w = width
      else
         w = nint(self%samprate / self%samprate_lowres)
      end if
      n    = int(ntod / w) + 1
      if (.not. present(tod_out)) then
         ext = [-npad, n+npad]
         return
      end if

      do i = 1, n-1
         j = (i-1)*w+1
         k = min(i*w,ntod)

         if (present(mask)) then
            tod_out(i) = sum(tod_in(j:k)*mask(j:k)) / sum(mask(j:k))
         else
            tod_out(i) = sum(tod_in(j:k)) / (k - j + 1)
         end if
         if (present(threshold)) then
            if (tod_out(i) <= threshold) then
               tod_out(i) = 0.
            else
               tod_out(i) = 1.
            end if
         end if
      end do
      if (present(threshold)) then
         tod_out(-npad:0)  = 0.
         tod_out(n:n+npad) = 0.

         ! Expand mask by m samples
         m = 1
         do i = 1, n ! Expand right edges
            if (tod_out(i) == 1 .and. tod_out(i+1) == 0) tod_out(i-m:i) = 0.
         end do
         do i = n, 1, -1 ! Expand left edges
            if (tod_out(i) == 1 .and. tod_out(i-1) == 0) tod_out(i:i+m) = 0.
         end do

      else
         tod_out(-npad:0)  = tod_out(1)
         tod_out(n:n+npad) = tod_out(n-1)
      end if

   !!$    if (self%myid == 0) then
   !!$       open(58, file='filter.dat')
   !!$       do i = 1, ntod
   !!$          write(58,*) i, tod_in(i)
   !!$       end do
   !!$       write(58,*)
   !!$       do i = -npad, n+npad
   !!$          write(58,*) i*astep, tod_out(i)
   !!$       end do
   !!$       close(58)
   !!$    end if
   !!$    call mpi_finalize(i)
   !!$    stop

   end subroutine downsample_tod



! Fills masked region with linear function between the mean of 20 points at each end
  subroutine fill_masked_region(d_p, mask, i_start, i_end, ntod, chunk)
    implicit none
    real(sp), intent(inout)  :: d_p(:)
    real(sp), intent(in)     :: mask(:)
    integer(i4b), intent(in) :: i_end, i_start, ntod, chunk
    real(dp)     :: mu1, mu2
    integer(i4b) :: i, n_mean, earliest, latest
    n_mean = 20
    earliest = max(i_start - (n_mean + 1), 1)
    latest = min(i_end + (n_mean + 1), ntod)
    if (i_start == 1) then  ! masked region at start for scan
       if (i_end < ntod) then
          mu2 = sum(d_p(i_end:latest) * mask(i_end:latest)) / sum(mask(i_end:latest))
          d_p(i_start:i_end) = mu2
       else
          !write(*,*) "Entirety of scan", chunk, "masked, this should not happen (in comm_tod_mod.fill_masked_region)"
          d_p(:) = 0.d0
          return
       end if
    else if (i_end == ntod) then  ! masked region at end of scan
       mu1 = sum(d_p(earliest:i_start) * mask(earliest:i_start)) / sum(mask(earliest:i_start))
       d_p(i_start:i_end) = mu1
    else   ! masked region in middle of scan
       mu1 = sum(d_p(earliest:i_start) * mask(earliest:i_start)) / sum(mask(earliest:i_start))
       mu2 = sum(d_p(i_end:latest) * mask(i_end:latest)) / sum(mask(i_end:latest))
       do i = i_start, i_end
          d_p(i) = mu1 + (mu2 - mu1) * (i - i_start + 1) / (i_end - i_start + 2)
       end do
    end if
  end subroutine fill_masked_region


! Identifies and fills masked region
  subroutine fill_all_masked(d_p, mask, ntod, sample, sigma_0, handle, chunk)
    implicit none
    real(sp),         intent(inout)  :: d_p(:)
    real(sp),         intent(in)     :: mask(:)
    real(sp),         intent(in), optional     :: sigma_0
    type(planck_rng), intent(inout), optional  :: handle
    integer(i4b),     intent(in) :: ntod
    logical(lgt),     intent(in) :: sample
    integer(i4b),     intent(in) :: chunk
    integer(i4b) :: j_end, j_start, j, k
    logical(lgt) :: init_masked_region, end_masked_region

    ! Fill gaps in data
    init_masked_region = .true.
    end_masked_region = .false.
    do j = 1,ntod
       if (mask(j) == 1.) then
          if (end_masked_region) then
             j_end = j - 1
             call fill_masked_region(d_p, mask, j_start, j_end, ntod, chunk)
             ! Add noise to masked region
             if (sample) then
                do k = j_start, j_end
                   d_p(k) = d_p(k) + sigma_0 * rand_gauss(handle)
                end do
             end if
             end_masked_region = .false.
             init_masked_region = .true.
          end if
       else
          if (init_masked_region) then
             init_masked_region = .false.
             end_masked_region = .true.
             j_start = j
          end if
       end if
    end do
    ! if the data ends with a masked region
    if (end_masked_region) then
       j_end = ntod
       call fill_masked_region(d_p, mask, j_start, j_end, ntod, chunk)
       if (sample) then
          do k = j_start, j_end
             d_p(k) = d_p(k) + sigma_0 * rand_gauss(handle)
          end do
       end if
    end if

  end subroutine fill_all_masked


  ! Compute chisquare
  subroutine compute_tod_chisq(self, scan, det, mask, s_sky, s_spur, &
       & n_corr, tod, s_jump, absbp, verbose)
    implicit none
    class(comm_tod),                 intent(inout)  :: self
    integer(i4b),                    intent(in)     :: scan, det
    real(sp),          dimension(:), intent(in)     :: mask, s_sky, s_spur
    real(sp),          dimension(:), intent(in)     :: n_corr
    real(sp),          dimension(:), intent(in)     :: tod
    real(sp),          dimension(:), intent(in), optional :: s_jump
    logical(lgt),                    intent(in), optional :: absbp, verbose

    
    real(dp)     :: chisq, d0, g
    integer(i4b) :: i, n

    call timer%start(TOD_CHISQ, self%band)

    chisq       = 0.d0
    n           = 0
    g           = self%scans(scan)%d(det)%gain
    do i = 1, self%scans(scan)%ntod
       if (mask(i) < 0.5) cycle
       n     = n+1
       d0    = tod(i) - (g * s_spur(i) + n_corr(i))
       if (present(s_jump)) d0 = d0 - s_jump(i)
       chisq = chisq + (d0 - g * s_sky(i))**2
    end do

    if (self%scans(scan)%d(det)%N_psd%sigma0 <= 0.d0) then
       if (present(absbp)) then
          self%scans(scan)%d(det)%chisq_prop   = 0.d0
       else
          self%scans(scan)%d(det)%chisq        = 0.d0
       end if
    else
       chisq      = chisq      / self%scans(scan)%d(det)%N_psd%sigma0**2
       if (present(absbp)) then
          self%scans(scan)%d(det)%chisq_prop   = chisq
       else
          self%scans(scan)%d(det)%chisq        = (chisq - n) / sqrt(2.d0*n)
       end if
    end if
    if (present(verbose)) then
      if (verbose) write(*,fmt='(a,i8,i3,e16.8,i10,f16.3)') "chi2 :  ", self%scanid(scan), det, &
         &self%scans(scan)%d(det)%N_psd%sigma0, n, self%scans(scan)%d(det)%chisq
    end if

    call timer%stop(TOD_CHISQ, self%band)

  end subroutine compute_tod_chisq

  function get_total_chisq(self, det)
    implicit none
    class(comm_tod),     intent(in)  :: self
    integer(i4b),        intent(in)  :: det
    real(dp)                         :: get_total_chisq

    integer(i4b) :: i, ierr
    real(dp)     :: chisq, buffer

    chisq = 0.d0
    do i = 1, self%nscan ! Sum chisq for all scans
       if (.not. self%scans(i)%d(det)%accept) cycle
       chisq = chisq + self%scans(i)%d(det)%chisq
    end do
    call mpi_reduce(chisq, buffer, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
         & 0, self%info%comm, ierr)

    get_total_chisq = buffer

  end function get_total_chisq

  subroutine symmetrize_flags(self, flag)
    implicit none
    class(comm_tod),                          intent(inout) :: self
    integer(i4b),         dimension(1:,1:),   intent(inout) :: flag

    integer(i4b) :: i, det

    do det = 1, self%ndet
       do i = 1, size(flag,1)
          if (iand(flag(i,det),self%flag0) .ne. 0) then
             flag(i,self%partner(det)) = flag(i,det)
          end if
       end do
    end do

  end subroutine symmetrize_flags

  subroutine decompress_pointing_and_flags(self, scan, det, pix, psi, flag)
    implicit none
    class(comm_tod),                    intent(in)            :: self
    integer(i4b),                       intent(in)            :: scan, det
    integer(i4b),        dimension(:),  intent(out), optional :: flag
    integer(i4b),        dimension(:,:),intent(out), optional :: psi, pix
    integer(i4b) :: i, j, k
    do i=1, self%nhorn
       if (present(pix)) then
          call huffman_decode2_int(self%scans(scan)%hkey, self%scans(scan)%d(det)%pix(i)%p,  pix(:,i))
       end if
       if (present(psi)) then
          call huffman_decode2_int(self%scans(scan)%hkey, self%scans(scan)%d(det)%psi(i)%p,  psi(:,i))
          if (minval(psi) < 1) then
            write(*,*) 'Psi bin ranges from ', minval(psi), maxval(psi), ', should be 1-indexed'
            stop
          end if
          if (maxval(psi) > self%npsi) then
            write(*,*) 'Psi bin ranges from ', minval(psi), maxval(psi), ', greater than npsi,', self%npsi
            stop
          end if
          if (self%polang(det) /= 0.) then
             do j = 1, size(psi,1)
                psi(j,i) = psi(j,i) + nint(self%polang(det)/(2.d0*pi)*self%npsi)
             end do
          end if
          do j = 1, size(psi,1)
             if (psi(j,i) < 1) then
                psi(j,i) = psi(j,i) + self%npsi
             else if (psi(j,i) > self%npsi) then
                psi(j,i) = psi(j,i) - self%npsi
             end if
          end do
       end if
    end do
    if (present(flag)) then
       call huffman_decode2_int(self%scans(scan)%hkey, self%scans(scan)%d(det)%flag, flag)

       ! Apply dynamic mask if it exists
       if (allocated(self%scans(scan)%d(det)%mask_dyn)) then
          do i = 1, size(self%scans(scan)%d(det)%mask_dyn,2)
             j = self%scans(scan)%d(det)%mask_dyn(1,i)
             k = self%scans(scan)%d(det)%mask_dyn(2,i)
             flag(j:k) = huge(flag(j))
          end do
       end if
    end if

  end subroutine decompress_pointing_and_flags

  subroutine decompress_diodes(self, scan, det, diodes, flag, pix, psi)
    ! Decompress per-diode tod information
    ! 
    ! Inputs:
    ! ----------
    ! self: comm_tod
    !
    ! scan: integer
    !     scan integer label
    ! det: integer
    !     detector number
    !
    ! Returns:
    ! --------
    ! diodes : real(sp) (ntod, ndiode)
    !    full raw diode values

    implicit none
    class(comm_tod),                    intent(in)  :: self
    integer(i4b),                       intent(in)  :: scan, det
    real(sp),          dimension(:,:),  intent(out) :: diodes
    integer(i4b),      dimension(:),    intent(out), optional :: flag
    integer(i4b),      dimension(:),    intent(out), optional :: pix
    integer(i4b),      dimension(:),    intent(out), optional :: psi

    integer(i4b) :: i, j
    real(sp)     :: tot
    integer(i4b), allocatable, dimension(:) :: buff

!    allocate(buff(size(diodes,1)))
    do i = 1, self%ndiode
!HKEHKE
        call huffman_decode2_sp(self%scans(scan)%todkey, self%scans(scan)%d(det)%zdiode(i)%p, diodes(:,i))
        !tot = sum(diodes(:,i))
        !call huffman_decode3(self%scans(scan)%todkey, self%scans(scan)%d(det)%zdiode(i)%p, buff)
!        write(*,*) sum(abs(diodes(:,i)-buff)), maxval(abs(diodes(:,i)-buff))
        !diodes(:,i) = buff
        !write(*,*) diodes(1:2,i), buff(1:2)
!!$        if (self%myid ==0) then
!!$           open(58,file='test2.dat')
!!$           do j = 1, size(buff)
!!$              write(58,*) diodes(j,i), buff(j)
!!$           end do
!!$           close(58)
!!$        end if
!!$        call mpi_finalize(j)
!!$        stop

    end do
!    deallocate(buff)

    if (present(flag)) then
       call huffman_decode2_int(self%scans(scan)%hkey, self%scans(scan)%d(det)%flag, flag)
    end if

    if (present(pix)) then ! this assumes nhorn = 1, sorry future person
      call huffman_decode2_int(self%scans(scan)%hkey, self%scans(scan)%d(det)%pix(1)%p, pix)
    end if

    if (present(psi)) then ! this assumes nhorn = 1, sorry future person
      call huffman_decode2_int(self%scans(scan)%hkey, self%scans(scan)%d(det)%psi(1)%p,  psi, imod=self%npsi-1)
      if (self%polang(det) /= 0.) then
         do j = 1, size(psi)
            psi(j) = psi(j) + nint(self%polang(det)/(2.d0*pi)*self%npsi)
            if (psi(j) < 1) then
               psi(j) = psi(j) + self%npsi
            else if (psi(j) > self%npsi) then
               psi(j) = psi(j) - self%npsi
            end if
         end do
      end if
   end if

  end subroutine decompress_diodes


  subroutine decompress_tod(self, scan, det, tod)
    !
    ! decompresses huffman coded tods 
    !
    ! Argumnets:
    ! ----------
    ! self: comm_tod
    !
    ! scan: integer
    !     scan integer label
    ! det: integer
    !     detector number
    !
    ! Returns:
    ! --------
    ! tod: real(sp)
    !    full raw TOD values
    !
    implicit none
    class(comm_tod),                    intent(in)  :: self
    integer(i4b),                       intent(in)  :: scan, det
    real(sp),            dimension(:),  intent(out) :: tod

    byte,  allocatable, dimension(:)  :: test

    call huffman_decode2_sp(self%scans(scan)%todkey, self%scans(scan)%d(det)%ztod, tod)

  end subroutine decompress_tod
  
  subroutine compute_solar_centered_pointing(tod, scan, det, pix, psi, pix_sol, psi_sol)
    implicit none
    class(comm_tod),                  intent(in)  :: tod
    integer(i4b),                     intent(in)  :: scan, det
    integer(i4b),        dimension(:),intent(in)  :: pix, psi
    integer(i4b),        dimension(:),intent(out) :: pix_sol, psi_sol

    integer(i4b) :: i, j
    real(dp)     :: alpha, lat, lon, vec(3), vec0(3), M_sun(3,3), x_sun(3), theta_sun, phi_sun, M_ecl2gal(3,3)

    call ecl_to_gal_rot_mat(M_ecl2gal)
    
    do j = 1, tod%scans(scan)%ntod
       ! Set up transformation from Ecliptic to Solar centered coordinates
       alpha = real(j-1,dp) / real(tod%scans(scan)%ntod-1,dp)
       x_sun = - (1.d0-alpha) * tod%scans(scan)%x0_obs - alpha * tod%scans(scan)%x1_obs ! Sun position at time t in Ecliptic coordinates
       call vec2ang(x_sun, theta_sun, phi_sun)
       call compute_euler_matrix_zyz(-phi_sun, 0.d0, 0.d0, M_sun)
       
       ! Compute pointing in solar centered coordinates
       call pix2vec_ring(tod%nside, pix(j), vec0) ! Galactic coordinates
       vec0 = matmul(transpose(M_ecl2gal), vec0)    ! Ecliptic coordinates
       vec0 = matmul(M_sun, vec0)                   ! Solar-centered coordinates
       call vec2pix_ring(tod%nside, vec0, pix_sol(j))
    end do
    psi_sol = 0 ! Not computed yet
    
  end subroutine compute_solar_centered_pointing


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Subroutine to save time-ordered-data chunk
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine write_tod_chunk(filename, tod)
    implicit none
    character(len=*),                   intent(in) :: filename
    real(sp),         dimension(:),     intent(in) :: tod
    ! Expects one-dimensional TOD chunk

    integer(i4b) :: unit, n_tod, t

    n_tod = size(tod)

    unit = getlun()
    open(unit,file=trim(filename), recl=1024)
    write(unit,*) '# TOD value in mK'
    do t = 1, n_tod
       write(unit,fmt='(e16.8)') tod(t)
    end do
    close(unit)
  end subroutine write_tod_chunk


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Subroutine to save map array to fits file 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine write_fits_file(filename, array, outmaps)
    implicit none
    character(len=*),                   intent(in) :: filename
    real(dp),         dimension(0:),    intent(in) :: array
    class(map_ptr),   dimension(:),     intent(inout) :: outmaps

    integer(i4b) :: np0, m

    do m = 0, size(array) - 1
       outmaps(1)%p%map(m, 1) = array(m)
    end do

    call outmaps(1)%p%writeFITS(filename)

  end subroutine write_fits_file

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Subroutine to save map array to fits file 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine write_fits_file_iqu(filename, array, outmaps)
    implicit none
    character(len=*),                    intent(in) :: filename
    real(dp),         dimension(0:, 1:), intent(in) :: array
    class(map_ptr),   dimension(:),      intent(inout) :: outmaps

    outmaps(1)%p%map = array

    call outmaps(1)%p%writeFITS(filename)

  end subroutine write_fits_file_iqu


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Generic deferred routines that do not do anything
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine diode2tod_inst(self, scan, map_sky, procmask, tod)
    ! 
    ! Generates detector-coadded TOD from low-level diode data
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_tod)
    !           TOD object
    ! scan:     int
    !           Scan ID number
    !
    ! Returns
    ! ----------
    ! tod:      ntod x ndet sp array
    !           Output detector TOD generated from raw diode data
    !
    implicit none
    class(comm_tod),                           intent(inout) :: self
    integer(i4b),                              intent(in)    :: scan
    real(sp),          dimension(0:,1:,1:,1:), intent(in)    :: map_sky
    real(sp),          dimension(:),           intent(in)    :: procmask
    real(sp),          dimension(:,:),         intent(out)   :: tod
    tod = 0.
  end subroutine diode2tod_inst

  subroutine read_tod_inst(self, file)
    ! 
    ! Reads instrument-specific common fields from TOD fileset
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_LB_tod)
    !           LB-specific TOD object
    ! file:     derived type (hdf_file)
    !           Already open HDF file handle; only root includes this
    !
    ! Returns
    ! ----------
    ! None, but updates self
    !
    implicit none
    class(comm_tod),                     intent(inout)          :: self
    type(hdf_file),                      intent(in),   optional :: file
  end subroutine read_tod_inst
  
  subroutine read_scan_inst(self, file, slabel, detlabels, scan)
    ! 
    ! Reads instrument-specific scan information from TOD fileset
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_tod)
    !           TOD object
    ! file:     derived type (hdf_file)
    !           Already open HDF file handle
    ! slabel:   string
    !           Scan label, e.g., "000001/"
    ! detlabels: string (array)
    !           Array of detector labels, e.g., ["27M", "27S"]
    ! scan:     derived class (comm_scan)
    !           Scan object
    !
    ! Returns
    ! ----------
    ! None, but updates scan object
    !
    implicit none
    class(comm_tod),                     intent(in)    :: self
    type(hdf_file),                      intent(in)    :: file
    character(len=*),                    intent(in)    :: slabel
    character(len=*), dimension(:),      intent(in)    :: detlabels
    class(comm_scan),                    intent(inout) :: scan
  end subroutine read_scan_inst

  subroutine initHDF_inst(self, chainfile, path)
    ! 
    ! Initializes instrument-specific TOD parameters from existing chain file
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_tod)
    !           TOD object
    ! chainfile: derived type (hdf_file)
    !           Already open HDF file handle to existing chainfile
    ! path:   string
    !           HDF path to current dataset, e.g., "000001/tod/030"
    !
    ! Returns
    ! ----------
    ! None
    !
    implicit none
    class(comm_tod),                 intent(inout)  :: self
    type(hdf_file),                      intent(in)     :: chainfile
    character(len=*),                    intent(in)     :: path
  end subroutine initHDF_inst

  subroutine load_instrument_inst(self, instfile, band)
    !
    ! Reads the instrument specific fields from the instrument file
    ! Implements comm_tod_mod::load_instrument_inst
    !
    ! Arguments:
    !
    ! self : comm_tod
    !    the tod object (this class)
    ! file : hdf_file
    !    the open file handle for the instrument file
    ! band : int
    !    the index of the current detector
    ! 
    ! Returns : None
    implicit none
    class(comm_tod),                     intent(inout) :: self
    type(hdf_file),                      intent(in)    :: instfile
    integer(i4b),                        intent(in)    :: band
  end subroutine load_instrument_inst

  
  subroutine dumpToHDF_inst(self, chainfile, path)
    ! 
    ! Writes instrument-specific TOD parameters to existing chain file
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_tod)
    !           TOD object
    ! chainfile: derived type (hdf_file)
    !           Already open HDF file handle to existing chainfile
    ! path:   string
    !           HDF path to current dataset, e.g., "000001/tod/030"
    !
    ! Returns
    ! ----------
    ! None
    !
    implicit none
    class(comm_tod),                     intent(in)     :: self
    type(hdf_file),                      intent(in)     :: chainfile
    character(len=*),                    intent(in)     :: path
  end subroutine dumpToHDF_inst

  subroutine remove_fixed_scans(self)
    ! 
    ! Sets accept = .false. for known bad scans
    ! 
    ! Arguments:
    ! ----------
    ! self:     derived class (comm_tod)
    !           TOD object
    !
    ! Returns
    ! ----------
    ! None
    !
    implicit none
    class(comm_tod),                     intent(inout)  :: self
  end subroutine remove_fixed_scans
  
  subroutine apply_map_precond(self, map, map_out)
    implicit none
    class(comm_tod),                   intent(in)    :: self
    real(dp),        dimension(0:,1:), intent(in)    :: map
    real(dp),        dimension(0:,1:), intent(out)   :: map_out

    map_out = map

  end subroutine apply_map_precond

  subroutine collect_v_sun(self)
    implicit none
    class(comm_tod),   intent(inout) :: self

    integer(i4b) :: i, j, ierr

    allocate(self%v_sun(3,self%nscan_tot))
    self%v_sun = 0.d0
    do i = 1, self%nscan
       self%v_sun(:,self%scanid(i)) = self%scans(i)%v_sun
    end do

    call mpi_allreduce(MPI_IN_PLACE, self%v_sun, size(self%v_sun), &
         & MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)

  end subroutine collect_v_sun

   subroutine precompute_zodi_lookups(self, cpar)
      class(comm_tod),   intent(inout) :: self
      type(comm_params),       intent(in) :: cpar

      integer(i4b) :: i, j, ierr, pix, pix_high, pix_low, nest_pix, n_subpix, nobs_lowres, npix_lowres, npix_highres
      real(dp), allocatable :: x0_obs(:, :), x1_obs(:, :), x0_earth(:, :), x1_earth(:, :), t0(:), t1(:), ind2vec_zodi_temp(:, :)
      real(dp), allocatable :: x0_obs_packed(:, :), x1_obs_packed(:, :), x0_earth_packed(:, :), x1_earth_packed(:, :), t0_packed(:), t1_packed(:)
      real(dp) :: r, obs_time_end, dt_tod, SECOND_TO_DAY, rotation_matrix(3, 3)
      real(dp), allocatable :: time(:), x_obs(:, :), x_earth(:, :)

      allocate(x0_obs(3, self%nscan_tot), x1_obs(3, self%nscan_tot))
      allocate(x0_earth(3, self%nscan_tot), x1_earth(3, self%nscan_tot))
      allocate(t0(self%nscan_tot), t1(self%nscan_tot))

      x0_obs = 0.
      x1_obs = 0.
      x0_earth = 0.
      x1_earth = 0.
      t0 = 0.
      t1 = 0.

      do i = 1, self%nscan
         t0(self%scanid(i)) = self%scans(i)%t0(1)
         t1(self%scanid(i)) = self%scans(i)%t1(1)
         x0_obs(:, self%scanid(i)) = self%scans(i)%x0_obs
         x1_obs(:, self%scanid(i)) = self%scans(i)%x1_obs
         x0_earth(:, self%scanid(i)) = self%scans(i)%x0_earth
         x1_earth(:, self%scanid(i)) = self%scans(i)%x1_earth
      end do

      
      call mpi_allreduce(MPI_IN_PLACE, t0, size(t0), MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)
      call mpi_allreduce(MPI_IN_PLACE, t1, size(t1), MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)
      call mpi_allreduce(MPI_IN_PLACE, x0_obs, size(x0_obs), MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)
      call mpi_allreduce(MPI_IN_PLACE, x1_obs, size(x1_obs), MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)
      call mpi_allreduce(MPI_IN_PLACE, x0_earth, size(x0_earth), MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)
      call mpi_allreduce(MPI_IN_PLACE, x1_earth, size(x1_earth), MPI_DOUBLE_PRECISION, MPI_SUM, self%comm, ierr)

      ! filter out non zero values
      t0_packed = pack(t0, t0 /= 0.)
      t1_packed = pack(t1, t1 /= 0.)
      if (size(t0_packed) /= size(t1_packed)) stop "t0 and t1 are not the same size"
  

      allocate(x0_obs_packed(3, size(t0_packed)), x1_obs_packed(3, size(t0_packed)))
      allocate(x0_earth_packed(3, size(t0_packed)), x1_earth_packed(3, size(t0_packed)))
      do i = 1, 3
         x0_obs_packed(i, :) = pack(x0_obs(i, :), x0_obs(i, :) /= 0.)
         x1_obs_packed(i, :) = pack(x1_obs(i, :), x1_obs(i, :) /= 0.)
         x0_earth_packed(i, :) = pack(x0_earth(i, :), x0_earth(i, :) /= 0.)
         x1_earth_packed(i, :) = pack(x1_earth(i, :), x1_earth(i, :) /= 0.)
      end do

      ! make new time, obs_pos and earth_pos arrays containing both chunk start and chunk end values
      allocate(time(size(t0_packed) * 2))
      allocate(x_obs(3, size(t0_packed) * 2))
      allocate(x_earth(3, size(t0_packed) * 2))
      do i = 1, size(t0_packed) * 2 , 2
         j = (i - 1) / 2 + 1 ! index from 1, to size(t0)
         time(i) = t0_packed(j)
         time(i + 1) = t1_packed(j)
         x_obs(:, i) = x0_obs_packed(:, j)
         x_obs(:, i + 1) = x1_obs_packed(:, j)
         x_earth(:, i) = x0_earth_packed(:, j)
         x_earth(:, i + 1) = x1_earth_packed(:, j)
      end do

      do i = 2, size(time)
         if (.not. time(i) > time(i - 1)) stop "precomputed MJD time array must be strictly increasing"
      end do

      do i = 1, 3
         call spline_simple(self%x_obs_spline(i), time, x_obs(i, :))
         call spline_simple(self%x_earth_spline(i), time, x_earth(i, :))
      end do

      self%zodi_init_cache_time = self%scans(1)%t0(1)
      call self%clear_zodi_cache()
      
      !allocate spectral quantities
      allocate(self%zodi_b_nu_spl_obj(self%ndet))

      ! allocate cache files and precompute ecliptic unit vectors
      call ecl_to_gal_rot_mat(rotation_matrix)
      allocate(self%zodi_scat_cache(self%nobs, self%zodi_n_comps, self%ndet))
      allocate(self%zodi_therm_cache(self%nobs, self%zodi_n_comps, self%ndet))
      self%zodi_scat_cache = -1.d0
      self%zodi_therm_cache = -1.d0
      allocate(self%ind2vec_ecl(3, self%nobs))
      do i = 1, self%nobs
         self%ind2vec_ecl(:,i) = matmul(self%ind2vec(:,i), rotation_matrix)
      end do
      
      ! If zodi sampling is turned on we precompute lowres zodi lookups
      if (.not. cpar%sample_zodi) return
      ! Skip if zodi nside = tod nside
      if (self%nside == zodi_nside) return
      n_subpix = (self%nside / zodi_nside)**2

      npix_lowres = 12*zodi_nside**2
      npix_highres = 12*self%nside**2

      ! Make lookup table for highres pixels to lowres pixels
      allocate(self%udgrade_pix_zodi(0:npix_highres - 1))
      do i = 0, npix_highres - 1
         call ring2nest(self%nside, i, nest_pix)
         nest_pix = nest_pix / n_subpix
         call nest2ring(zodi_nside, nest_pix, self%udgrade_pix_zodi(i))
      end do
   
   end subroutine precompute_zodi_lookups

   subroutine clear_zodi_cache(self, obs_time)
      ! Clears the zodi tod cache used to look up already computed zodi values. 
      !
      ! This cache has an associate time of observation since the cache is only valid
      ! if the time between the observations are small enough for the observer to not 
      ! have moved significantly.
      class(comm_tod),   intent(inout) :: self
      real(dp), intent(in), optional :: obs_time

      self%zodi_scat_cache = -1.d0
      self%zodi_therm_cache = -1.d0
      if (present(obs_time)) then
         self%zodi_cache_time = obs_time
      else 
         self%zodi_cache_time = self%zodi_init_cache_time
      end if
      if (allocated(self%zodi_therm_cache_lowres)) then
         self%zodi_scat_cache_lowres = -1.d0
         self%zodi_therm_cache_lowres = -1.d0
      end if
   end subroutine clear_zodi_cache

   subroutine create_dynamic_mask(self, scan, det, res, rms_range, mask)
     implicit none
     class(comm_tod),                   intent(inout) :: self
     integer(i4b),                      intent(in)    :: scan, det
     real(sp),            dimension(:), intent(in)    :: res
     real(sp),            dimension(2), intent(in)    :: rms_range
     real(sp),            dimension(:), intent(inout) :: mask
     
     logical(lgt) :: cut
     integer(i4b) :: i, j, k, n, ntod, nmax
     real(dp) :: box_width, rms, vec(3), elon
     integer(i4b), allocatable, dimension(:,:) :: bad, buffer
     !real(dp),     allocatable, dimension(:,:) :: mask_solar
     
     ntod = size(res)
     nmax = 1000
     
      ! Compute rms
      rms = 0.d0
      n   = 0
      do i = 1, ntod
         if (mask(i) /= 1.) cycle
         rms = rms + res(i)**2
         n   = n   + 1
      end do
      rms = sqrt(rms/(n-1))

!      write(*,*) 'a'
      ! Get full-sky mask
!      if (associated(self%mask_solar)) then
!         allocate(mask_solar(0:12*self%nside**2-1,1))
!         call self%mask_solar%bcast_fullsky_map(mask_solar)
!      end if

      
!      write(*,*) 'b'
      ! Look for strong outliers and masked samples, save bad ranges
      allocate(bad(2,nmax))
      bad = -1
      n   = 0
      do i = 1, ntod

         ! Apply RMS selection criterium
         if (mask(i) == 1) then
            cut = res(i) < rms_range(1)*rms .or. res(i) > rms_range(2)*rms
         else
            cut = .false.
         end if
         
         ! Apply solar mask selection criterium
!         call pix2vec_ring(self%nside, self%scans(scan)%d(det)%pix_sol(i,1), vec)
!         elon = acos(min(max(vec(1),-1.d0),1.d0)) * 180.d0/pi                       ! The Sun is at (1,0,0)
         !         cut = cut .or. elon < self%sol_elong_range(1) .or. elon > self%sol_elong_range(2)
         if (allocated(self%mask_solar)) then
            cut = cut .or. (self%mask_solar(self%scans(scan)%d(det)%pix_sol(i,1),1) < 0.5)
         end if

         if (cut) then
            ! Start new range if not already active
            if (bad(1,n+1) == -1) bad(1,n+1) = i
            mask(i) = 0.
         else
            ! Close active range
            if (bad(1,n+1) /= -1 .and. bad(2,n+1) == -1) then
               bad(2,n+1) = i-1
               n          = n+1
            end if
         end if
         
         ! Increase array size if needed
         if (n == nmax) then
            nmax = 2*nmax
            allocate(buffer(2,nmax))
            buffer = -1
            buffer(:,1:nmax/2) = bad
            deallocate(bad)
            allocate(bad(2,nmax))
            bad = buffer
            deallocate(buffer)
         end if
      end do

      ! Close open range if needed at the end
      if (bad(1,n+1) /= -1 .and. bad(2,n+1) == -1) then
         bad(2,n+1) = ntod
         n          = n+1
      end if
      
      ! Store final array
      if (n > 0) then
         allocate(self%scans(scan)%d(det)%mask_dyn(2,n))
         self%scans(scan)%d(det)%mask_dyn = bad(:,1:n)
!!$         do i = 1, n
!!$            write(*,*) i, bad(:,i)
!!$         end do
         !write(*,fmt='(a,i6,a,i6,i4)') ' Removing ', n, ' ranges in dynamic mask for scan, det', self%scanid(scan), det
      end if

      
!      write(*,*) 'c'
      deallocate(bad)
!      if (allocated(mask_solar)) deallocate(mask_solar)
   end subroutine

   subroutine distribute_sky_maps(tod, map_in, scale, map_out, map_full)
    implicit none
    class(comm_tod),                       intent(in)     :: tod
    type(map_ptr), dimension(1:,1:),       intent(inout)  :: map_in       ! (ndet,ndelta)    
    real(sp),                              intent(in)     :: scale
    real(sp),      dimension(1:,1:,0:,1:), intent(out)    :: map_out
    real(dp),      dimension(1:, 0:), intent(out), optional   :: map_full

    integer(i4b) :: i, j, k, l, npix, nmaps
    real(dp),     allocatable, dimension(:,:) :: m_buf

    npix  = map_in(1,1)%p%info%npix
    nmaps = map_in(1,1)%p%info%nmaps
    allocate(m_buf(0:npix-1,nmaps))
    if (present(map_full)) map_full = 0
    do j = 1, size(map_in,2)       ! ndelta
       do i = 1, size(map_in,1)    ! ndet
          map_in(i,j)%p%map = scale * map_in(i,j)%p%map ! unit conversion
          call map_in(i,j)%p%bcast_fullsky_map(m_buf)
          do k = 1, tod%nobs
             map_out(:,k,i,j) = m_buf(tod%ind2pix(k),:)
          end do
          if (present(map_full) .and. j .eq. 1) then
            do k = 1, nmaps
              do l = 0, npix-1
                map_full(k, l) = map_full(k, l) + m_buf(l, k)
              end do
            end do
          end if
       end do
       do k = 1, tod%nobs
          do l = 1, tod%nmaps
             map_out(l,k,0,j) = sum(map_out(l,k,1:tod%ndet,j))/tod%ndet
          end do
       end do
    end do

    if (present(map_full)) map_full = map_full/tod%ndet

    deallocate(m_buf)

  end subroutine distribute_sky_maps


  subroutine get_s_static(self, band, point, s)
     ! Evaluates the solar centric TOD
     !
     ! Parameters:
     ! -----------
     ! self  : ZodiModel object
     ! band  : Band number id
     ! point : pointing array
     ! s     : Output TOD of static zodi signal
     implicit none
     class(comm_tod),            intent(in)  :: self
     integer(i4b),               intent(in)  :: band
     integer(i4b), dimension(:), intent(in)  :: point
     real(sp),     dimension(:), intent(out) :: s

     integer(i4b) :: i

     if (.not. associated(self%map_solar)) then
        s = 0.d0
        return
     end if
     
     do i = 1, size(s)
        s(i) = self%map_solar(point(i),1)
     end do

   end subroutine get_s_static

  
   
end module comm_tod_mod
