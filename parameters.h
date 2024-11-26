// Copyright (c) 2024 Jie Jiang (江捷) <jiejiang@pusan.ac.kr>
// This code is licensed under the MIT License.
// See the LICENSE file in the project root for license information.

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cmath>

#define VV 2
#define NDIMS 3
#define SPECTRAL_FLAG true
#define PAD_FLAG false
#define SNAPSHOTS_ENERGY true
#define WITH_GW false
#define BIFURCATION false
#define SIMULATE_INFLATION true
#define TEST true

const int nflds = 1;  //Number of fields
const int n_threads = 1; // 选择合理的线程数量
const int N =  64; // 64; // 32; //Number of points along each edge of the cubical lattice
const int N_pad = 128;

const int maxnumbins = (int)(sqrt((double)NDIMS) * (N/2)) + 2;
// constexpr size_t maxnumbins = (size_t)(std::sqrt((double)NDIMS) * (N/2)) + 2;

const double kIR = 25; //0.375; // 0.75; // 
const int kIR_prec = 1;
const double a0 = 1.;
const double kcutoff = 0; // Momentum for initial lowpass filter. Set to 0 to not filter
const double L = 2 * M_PI / kIR; // Size of box (i.e. length of each edge) in rescaled distance units
const double dx = L / (double)N; // Distance between adjacent gridpoints

// const double dt = 1.e-3; // Size of time step
extern double dt;


const double epsilon = 1.e-3;

// const bool analytic = false;
#define ANALYTIC_EVOLUTION false
const double t_analytic = 6; // Final time
// const double dt_analytic = 1.e-5;// 1.e-6; // Time step
extern double dt_analytic;

const double tcheck = 7.2;// 1.2; // Final time
const double tf = 9; // 3; // Final time

const double tOutputFreq  = 0.001;
const int Freq_prec = 3;
const int setw_leng = 15;

const double tOutputInfreq = 0.01;
const int Infreq_prec = 2;

const bool ssnapshot = false;
const double tSnapFreq = 0.05;

const std::string ext_ = ".txt";
const std::string dir_ = "output";
const std::string snap_dir_ = "/Snapshots";

const int seed = 1;
const int expansion = 2;
const double expansion_power = .5;


const bool continue_run = false;

const std::string alt_extension = ".txt";

const double checkpoint_interval = 10.;
const double store_lattice_times[] = {0.};
const bool smeansvars = true;
const int sexpansion = 1;

const bool smodel = false;
const double t_start_output = 0.;

const bool scheckpoint = false;
const double tcheckpoint = t_start_output;

const bool sspectra = true;
const double tspectra = t_start_output;

const bool senergy = true;
const double tenergy = t_start_output;

const bool shistograms = true;
const double thistograms = t_start_output;
const int nbins = 256; // Number of bins
const double histogram_min = 0., histogram_max = 0.;

const bool shistograms2d = false; // Output two dimensional histograms of fields
const double thistograms2d = t_start_output;
const int nbins2d = 10, nbins0 = nbins2d, nbins1 = nbins2d; // Number of bins in each field direction
const double histogram2d_min = 0., histogram2d_max = 0.; // Upper and lower limits of the histograms. To use all current field values set these two to be equal.
const int hist2dflds[] = {0, 1}; // Pairs of fields to be evaluated. This array should always contain an even number of integers.

const bool sslices = false; // Output the field values on a slice through the lattice
const double tslices = t_start_output;
const int slicedim = 1; // Dimensions of slice to be output. (If slicedim>=NDIMS the whole lattice will be output.) Warning: If slicedim=3 the resulting file may be very large.
const int slicelength = N, sliceskip = 1; // The slices will use every <sliceskip> point up to a total of <slicelength>.  Set length=N and skip=1 to output all points in the slice.
const int sliceaverage = 1; 


// const bool strajectory = false;


const double m_in_MPl = 1.e-6;
const double alpha_in_PHYS = 150;
const double beta_0 = 1.4;
const double phi_b = 5.;
const double chi_b = 0; // 2.e-18; // 1.e-15; // 
const double sigma_1 = 0.04;
const double sigma_2 = 0.0001;



const int alpha = 0; // conformal time; alpha = 0, cosmic time
const double MPl = 1;

const double initfield[] = {11. * MPl}; // {16. * MPl,0}; // pgf = f / fStar, Initial values of the fields in program units. All nonspecified values are taken to be zero.
const double initderivs[] = {-0.815284e-6 * MPl * MPl}; // pgfd = fd / fStar / omegaStar, Initial values of the field derivatives in program units. All nonspecified values are taken to be zero.


const double fStar = MPl;
const double omegaStar = m_in_MPl * fStar;

#if SIMULATE_INFLATION
// const double DeltaF_DeltaPi = pow(omegaStar / fStar, 2) * pow(N / dx, 3) / 2.;
const double DeltaF_DeltaPi = pow(omegaStar / fStar, 2) * (N / dx) / 2.;
#endif



#endif // PARAMETERS_H
