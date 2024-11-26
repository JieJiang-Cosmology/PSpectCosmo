// Copyright (c) 2024 Jie Jiang (江捷) <jiejiang@pusan.ac.kr>
// This code is licensed under the MIT License.
// See the LICENSE file in the project root for license information.

#ifndef _LATTICEEASYHEADER_
#define _LATTICEEASYHEADER_

#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <H5Cpp.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <unordered_map>
#include <filesystem>
#include <set>
#include <algorithm>

#include "parameters.h"

const double pi = (double)(2.*asin(1.));
inline double pw2(double x) {return x * x;} // Useful macro for squaring doubles



/////////////////////////////////INCLUDE ADJUSTABLE PARAMETERS///////////////////

using Vector3 = std::array<double, 3>;

using Matrix3x3 = std::array<std::array<double, 3>, 3>;

using ComplexMatrix3x3 = std::array<std::array<std::complex<double>, 3>, 3>;

using Array3D = std::array<std::array<std::array<double, N>, N>, N>;

using Array4D = std::array<std::array<std::array<std::array<double, N>, N>, N>, nflds>;

/////////////////////////////////GLOBAL DYNAMIC VARIABLES////////////////////////
extern double t, t0; // Current time and initial time (t0=0 unless the run is a continuation of a previous one)
extern double a, ad, add; // Scale factor and its derivatives (aterm is a combination of the others used in the equations of motion)

extern double hubble_init; // Initial value of the Hubble constant
extern int run_number; 


extern double EK_0[nflds];
extern double EG_0[nflds];

extern double EK_0_tot;
extern double EG_0_tot;

// extern std::ofstream trajectory_[N][N][N];
extern size_t numsteps;

/////////////////////////////////NON-ADJUSTABLE VARIABLES////////////////////////

/////////////////////////////////DIMENSIONAL SPECIFICATIONS//////////////////////
extern double f[nflds][N][N][N], pi_f[nflds][N][N][N];


extern double data_1st_deri[N][N][N];
extern double grad_energy[N][N][N];
extern double kine_energy[N][N][N];
extern double pot_energy[N][N][N];
extern double tot_energy[N][N][N];


extern fftw_complex *f_k[nflds];
extern fftw_complex *pi_k[nflds];
extern fftw_complex *Cache_Data_k;

#if !SPECTRAL_FLAG
    extern double pi_half[nflds][N][N][N];
    extern double f_laplacian[nflds][N][N][N];
#else
    extern fftw_complex* pi_k_half[nflds];
    extern double V_prime[nflds][N][N][N];
    extern fftw_complex* V_prime_k[nflds];
    extern fftw_complex* kernel_f_k[nflds];
    #if PAD_FLAG
        extern fftw_complex* Cache_Data_k_pad;
        extern double f_pad[nflds][N_pad][N_pad][N_pad];
        extern fftw_complex* f_k_pad[nflds];
        extern double V_prime_pad[nflds][N_pad][N_pad][N_pad];
        extern fftw_complex* V_prime_k_pad[nflds];
        extern fftw_plan plan_pad;
    #endif
#endif

extern fftw_plan plan;
extern fftw_plan plan_f, plan_pi;



#if WITH_GW


extern double u[3][3][N][N][N], u_lapl[3][3][N][N][N], pi_u_half[3][3][N][N][N];
extern fftw_complex *u_k[3][3];

extern double f_derivative[nflds][3][N][N][N];

extern double pi_u[3][3][N][N][N];
extern fftw_complex *pi_u_k[3][3];

extern double V_prime[nflds][N][N][N];
extern double V_prime_pad[nflds][N_pad][N_pad][N_pad];

extern bool DerivateIsCalculated;
#endif




extern double R[N][N][N];
extern fftw_complex *R_k;

extern double vacuum_energy;


#if ANALYTIC_EVOLUTION
extern std::vector<std::vector<std::complex<double>>> fk_left;
extern std::vector<std::vector<std::complex<double>>> pi_k_left;

extern std::vector<std::vector<std::complex<double>>> fk_right;
extern std::vector<std::vector<std::complex<double>>> pi_k_right;

extern std::set<int> k_square;
extern std::vector<int> k_square_vec;

extern double a_k, ad_k, add_k;
extern size_t num_k_values;
#endif





#if BIFURCATION
extern double f_positive[nflds][N][N][N];
extern double f_negative[nflds][N][N][N];
extern double f_av_positive[nflds];
extern double f_av_negative[nflds];

// // 声明全局变量
extern std::vector<std::vector<std::complex<double>>> fk_positive;
extern std::vector<std::vector<std::complex<double>>> pi_k_positive;
extern std::vector<std::vector<std::complex<double>>> fk_negative;
extern std::vector<std::vector<std::complex<double>>> pi_k_negative;


// extern std::array<std::array<std::complex<double>, maxnumbins>, nflds> fk_positive;
// extern std::array<std::array<std::complex<double>, maxnumbins>, nflds> pi_k_positive;
// extern std::array<std::array<std::complex<double>, maxnumbins>, nflds> fk_negative;
// extern std::array<std::array<std::complex<double>, maxnumbins>, nflds> pi_k_negative;


extern double f_initial[nflds][N][N][N];
extern double pi_initial[nflds][N][N][N];
extern fftw_complex* f_k_initial[nflds];
extern fftw_complex* pi_k_initial[nflds];

extern int chi_bifurcation_sign[N][N][N];
extern int count_positive;

#endif

#if SIMULATE_INFLATION
extern double R_Q2C[nflds][N][N][N/2+1];
#endif
extern double K_E, G_E, V_E;


const int N3 = N * N * N; // Number of spatial points in the grid
const int N3_pad = N_pad * N_pad * N_pad; // Number of spatial points in the grid

extern bool isChanged;

#define FIELD(fld) f[fld][i][j][k]
// #define FIELDD(fld) fd[fld][i][j][k]
#define PI_FIELD(fld) pi_f[fld][i][j][k]
#define FIELDPOINT(fld,i,j,k) f[fld][i][j][k]
#define LOOP for(i = 0; i < N; i++) for(j = 0; j < N; j++) for(k = 0; k < N; k++)
#define LOOP_k for(i = 0; i < N; i++) for(j = 0; j < N; j++) for(k = 0; k < N/2+1; k++)
#define LOOP_pad for(i = 0; i < N_pad; i++) for(j = 0; j < N_pad; j++) for(k = 0; k < N_pad; k++)
#define INDEXLIST int i, int j, int k
#define DECLARE_INDICES int i, j, k;


/////////////////////////////////INCLUDE SPECIFIC MODEL//////////////////////////
#include "model.h"
void print_memory_usage();


/////////////////////////////////FUNCTION DECLARATIONS///////////////////////////
///////////////////////// initialize.cpp /////////////////////////
void checkdirectory();
void initialize(); // Set initial parameters and field values
#if ANALYTIC_EVOLUTION
    void initialize_analytic_perturbation();
    void initialize_lattice();
    void k_square_initialize();
#endif


#if BIFURCATION
void initialize_k();
#endif


#if WITH_GW
double norm(const Vector3 &v);
void initialize_GW();
#endif


///////////////////////// evolution.cpp /////////////////////////

#if ANALYTIC_EVOLUTION
    void evolve_VV_analytic_perturbation(double dt);
    double kernel_a_only_backgroud(const double a, double (&f)[nflds], const double (&pi_f)[nflds]);
#endif

#if !SPECTRAL_FLAG
    double kernel_a(void);
    void evolve_VV(double dt);
    double gradient_energy(size_t fld);
    double kinetic_energy(size_t fld);
    #if BIFURCATION
        void evolve_VV_with_perturbation(double dt);
    #endif
#else
    double kinetic_energy_k(size_t fld);
    double gradient_energy_k(size_t fld);
    #if !PAD_FLAG
        double kernel_a_k_nopad(void);
        void evolve_VV_k_nopad(double dt);
    #else 
        void evolve_VV_k_pad(double dt);
        void padding(fftw_complex* f_k, fftw_complex* f_k_pad);
        double kernel_a_k_pad(const double a, double (&f)[nflds][N][N][N], double (&f_pad)[nflds][N_pad][N_pad][N_pad], fftw_complex** pi_k);
    #endif
#endif

void kinetic_energy_lattice(double a, const double (&pi_f)[nflds][N][N][N]);
void gradient_energy_lattice(double a, double (&f)[nflds][N][N][N]);

void evolve_VV_k_nopad_with_perturbation(double dt);
void bifurcation_sign();

double kernel_a_only_backgroud(const double a, double (&f)[nflds], const double (&pi_f)[nflds]);

///////////////////////// output.cpp /////////////////////////

void output_parameters();


#if ANALYTIC_EVOLUTION
void save_analytic();
void save_Freq_analytic();
void save_Infreq_analytic();
#endif

void save();
void save_Freq();
void save_Infreq();
void save_Snap(double);

#if BIFURCATION
void meansvars_bifurcation();
void d2Vdf2_evolution();
void spectra_trajectory_analytic();
void spectra_trajectory_lattice();
#endif


///////////////////////// Algorithm /////////////////////////
#if VV == 2
    const size_t s = 1;
    const double w[s] = {1.};
#elif VV == 4
    const size_t s = 3;
    const double w[s] = {1.351207191959657771818, -1.702414403875838200264, 1.351207191959657771818};
#elif VV == 6
    const size_t s = 7;
    const double w[s] = {0.78451361047755726382, 0.23557321335935813368, -1.1776799841788710069, 1.3151863206839112189, -1.1776799841788710069, 0.23557321335935813368, 0.78451361047755726382};
#elif VV == 8
    const size_t s = 15;
    const double w[s] = {0.74167036435061295345, -0.40910082580003159400,  0.19075471029623837995, -0.57386247111608226666,  0.29906418130365592384,  0.33462491824529818378, 0.31529309239676659663, -0.79688793935291635402,  0.31529309239676659663, 0.33462491824529818378, 0.29906418130365592384, -0.57386247111608226666, 0.19075471029623837995, -0.40910082580003159400, 0.74167036435061295345};
#elif VV == 10
    const size_t s = 31;
    const double w[s] = {-0.48159895600253002870, 0.0036303931544595926879, 0.50180317558723140279, 0.28298402624506254868, 0.80702967895372223806, -0.026090580538592205447, -0.87286590146318071547, -0.52373568062510581643, 0.44521844299952789252, 0.18612289547097907887, 0.23137327866438360633, -0.52191036590418628905, 0.74866113714499296793, 0.066736511890604057532, -0.80360324375670830316, 0.91249037635867994571, -0.80360324375670830316, 0.066736511890604057532, 0.74866113714499296793, -0.52191036590418628905, 0.23137327866438360633, 0.18612289547097907887, 0.44521844299952789252, -0.52373568062510581643, -0.87286590146318071547, -0.026090580538592205447, 0.80702967895372223806, 0.28298402624506254868, 0.50180317558723140279, 0.0036303931544595926879, -0.48159895600253002870};
#endif


#endif // End of conditional for definition of _LATTICEEASYHEADER_ macro
