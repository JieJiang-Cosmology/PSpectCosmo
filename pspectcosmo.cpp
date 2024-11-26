/*
LATTICEEASY consists of the C++ files ``latticeeasy.cpp,''
``initialize.cpp,'' ``evolution.cpp,'' ``output.cpp,''
``latticeeasy.h,'' ``parameters.h,''. (The distribution also includes
the file ffteasy.cpp but this file is distributed separately and
therefore not considered part of the LATTICEEASY distribution in what
follows.) LATTICEEASY is free. We are not in any way, shape, or form
expecting to make money off of these routines. We wrote them for the
sake of doing good science and we're putting them out on the Internet
in case other people might find them useful. Feel free to download
them, incorporate them into your code, modify them, translate the
comment lines into Swahili, or whatever else you want. What we do want
is the following:
1) Leave this notice (i.e. this entire paragraph beginning with
``LATTICEEASY consists of...'' and ending with our email addresses) in
with the code wherever you put it. Even if you're just using it
in-house in your department, business, or wherever else we would like
these credits to remain with it. This is partly so that people can...
2) Give us feedback. Did LATTICEEASY work great for you and help
your work?  Did you hate it? Did you find a way to improve it, or
translate it into another programming language? Whatever the case
might be, we would love to hear about it. Please let us know at the
email address below.
3) Finally, insofar as we have the legal right to do so we forbid
you to make money off of this code without our consent. In other words
if you want to publish these functions in a book or bundle them into
commercial software or anything like that contact us about it
first. We'll probably say yes, but we would like to reserve that
right.

For any comments or questions you can reach us at
gfelder@email.smith.edu
Igor.Tkachev@cern.ch

Enjoy LATTICEEASY!

Gary Felder and Igor Tkachev
*/


// Copyright (c) 2024 Jie Jiang (江捷) <jiejiang@pusan.ac.kr>
// This code is licensed under the MIT License.
// See the LICENSE file in the project root for license information.

#include "pspectcosmo.h"

double f[nflds][N][N][N], pi_f[nflds][N][N][N];

double data_1st_deri[N][N][N];
double grad_energy[N][N][N];
double kine_energy[N][N][N];
double pot_energy[N][N][N];
double tot_energy[N][N][N];

fftw_complex *f_k[nflds];
fftw_complex *pi_k[nflds];
fftw_complex *Cache_Data_k;

#if !SPECTRAL_FLAG
    double pi_half[nflds][N][N][N];
    double f_laplacian[nflds][N][N][N];
#else
    fftw_complex* pi_k_half[nflds];
    double V_prime[nflds][N][N][N];
    fftw_complex* V_prime_k[nflds];
    fftw_complex* kernel_f_k[nflds];
    #if PAD_FLAG
        fftw_complex* Cache_Data_k_pad;
        double f_pad[nflds][N_pad][N_pad][N_pad];
        fftw_complex* f_k_pad[nflds];
        double V_prime_pad[nflds][N_pad][N_pad][N_pad];
        fftw_complex* V_prime_k_pad[nflds];
        fftw_plan plan_pad;
    #endif
#endif

#if WITH_GW

double u[3][3][N][N][N], u_lapl[3][3][N][N][N], pi_u_half[3][3][N][N][N];
fftw_complex *u_k[3][3];

double f_derivative[nflds][3][N][N][N];

double pi_u[3][3][N][N][N];
fftw_complex *pi_u_k[3][3];

bool DerivateIsCalculated;

#endif


// double dt = 1.e-7;
// double dt_analytic = 1e-7;


double dt;
double dt_analytic;

double R[N][N][N];
fftw_complex *R_k;




#if ANALYTIC_EVOLUTION
std::vector<std::vector<std::complex<double>>> fk_left;
std::vector<std::vector<std::complex<double>>> pi_k_left;

std::vector<std::vector<std::complex<double>>> fk_right;
std::vector<std::vector<std::complex<double>>> pi_k_right;

std::set<int> k_square;
std::vector<int> k_square_vec;

double a_k = a0, ad_k = 0., add_k = 0.;
size_t num_k_values;
#endif



#if BIFURCATION
double f_positive[nflds][N][N][N];
double f_negative[nflds][N][N][N];
double f_av_positive[nflds];
double f_av_negative[nflds];




std::vector<std::vector<std::complex<double>>> fk_positive(nflds, std::vector<std::complex<double>>(maxnumbins));
std::vector<std::vector<std::complex<double>>> pi_k_positive(nflds, std::vector<std::complex<double>>(maxnumbins));
std::vector<std::vector<std::complex<double>>> fk_negative(nflds, std::vector<std::complex<double>>(maxnumbins));
std::vector<std::vector<std::complex<double>>> pi_k_negative(nflds, std::vector<std::complex<double>>(maxnumbins));



double f_initial[nflds][N][N][N];
double pi_initial[nflds][N][N][N];
fftw_complex* f_k_initial[nflds];
fftw_complex* pi_k_initial[nflds];

double K_E, G_E, V_E;

int chi_bifurcation_sign[N][N][N];
int count_positive;
#endif



#if SIMULATE_INFLATION
double R_Q2C[nflds][N][N][N/2+1];
#endif

double t, t0;
double a = a0, ad = 0., add = 0.;
double hubble_init = 0.;
int run_number = 1; // 0 for a first run, 1 for a continuation of a "0" run, etc.. Stored in the grid image (see checkpoint() function).



fftw_plan plan;
fftw_plan plan_f;
fftw_plan plan_pi;


size_t numsteps = 0;
bool isChanged;


double EK_0[nflds];
double EG_0[nflds];



double EK_0_tot = 0.;
double EG_0_tot = 0.;

double vacuum_energy;

// void print_memory_usage() {
//     struct rusage usage;
//     getrusage(RUSAGE_SELF, &usage);
//     std::cout << "Memory usage: " << usage.ru_maxrss << " KB" << std::endl;
// }


bool containsNaN(const std::vector<std::vector<std::complex<double>>>& vec, std::string_view name) {
    return std::any_of(vec.begin(), vec.end(), [&name](const auto& inner_vec) {
        return std::any_of(inner_vec.begin(), inner_vec.end(), [&name](const std::complex<double>& c) {
            if (std::isnan(c.real()) || std::isnan(c.imag())) {
                std::cerr << "NaN value detected at " << name << std::endl;
                return true;
            }
            return false;
        });
    });
}

// 主函数：检查所有四个变量
bool checkAllVectors(
    const std::vector<std::vector<std::complex<double>>>& fk_left,
    const std::vector<std::vector<std::complex<double>>>& pi_k_left,
    const std::vector<std::vector<std::complex<double>>>& fk_right,
    const std::vector<std::vector<std::complex<double>>>& pi_k_right
) {
    return containsNaN(fk_left, "fk_left") ||
           containsNaN(pi_k_left, "pi_k_left") ||
           containsNaN(fk_right, "fk_right") ||
           containsNaN(pi_k_right, "pi_k_right");
}


double calculate_dt(double k, double a) {
    int factor = 10; 
    double result = 2.0 / (k / a) / factor;

    // int exponent = std::floor(std::log10(std::abs(result)));
    // double mantissa = result / std::pow(10, exponent);

    // double dt;
    // if (mantissa > 5) {
    //     dt = 5 * std::pow(10, exponent);
    // } else if (mantissa >= 2.5 && mantissa <= 5) {
    //     dt = 2.5 * std::pow(10, exponent);
    // } else if (mantissa >= 2 && mantissa < 2.5) {
    //     dt = 2 * std::pow(10, exponent);
    // } else {
    //     dt = 1 * std::pow(10, exponent);
    // }

    // return dt;
    return result;
}


int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;

    checkdirectory();


    #if SPECTRAL_FLAG && PAD_FLAG
        if (N_pad <= N) {
            std::cerr << "You are using padding, N_pad must larger than N. Please reset the parameters!" << std::endl;
            return 1;
        }
    #endif

    fftw_init_threads();
    fftw_plan_with_nthreads(n_threads);

    Cache_Data_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));
    #if SPECTRAL_FLAG && PAD_FLAG
        Cache_Data_k_pad = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
    #endif

    R_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));

    for (size_t fld = 0; fld < nflds; fld++) {
        f_k[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));
        pi_k[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));

        #if SPECTRAL_FLAG
            pi_k_half[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));
            V_prime_k[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));
            kernel_f_k[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));
            #if PAD_FLAG
                f_k_pad[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
                V_prime_k_pad[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
            #endif
        #endif

        #if WITH_GW
            for (size_t i = 0; i < 3; i++)
                for (size_t j = 0; j < 3; j++) {
                    u_k[i][j] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));
                    pi_u_k[i][j] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));
                }
        #endif

    }

    // std::string filename = dir_ + "/output.txt";
    // std::ofstream output_(filename);

    // if (!output_) {
    //     std::cerr << "Failed to open file: " << filename << std::endl;
    //     return 1;
    // }


    if (seed < 1) // The use of seed<1 turns off certain functions (random numbers, fourier transforms, gradients, and potential energy) and should only be used for debugging
        std::cout << "Warning: The parameter seed has been set to " << seed << ", which will result in incorrect output. For correct output set seed to a positive integer.";

    t0 = 0.;
    t = t0;

    a = a0;
    // a_k = a0;

    // dt_analytic = calculate_dt(kIR * N * sqrt(3) / 2., a_k);


    size_t OutputFreq_interval_analytic;
    size_t OutputInfreq_interval_analytic;
    size_t Step_inteval;

    if (dt_analytic > tOutputFreq) {
        std::cerr << "dt_analytic should not larger than tOutputFreq: " << std::endl;
        std::exit(1);
    } else {
        OutputFreq_interval_analytic = static_cast<size_t>(tOutputFreq / dt_analytic + 0.5);
    }

    if (dt_analytic > tOutputInfreq) {
        std::cerr << "dt_analytic should not larger than tOutputInfreq: " << std::endl;
        std::exit(1);
    } else {
        OutputInfreq_interval_analytic = static_cast<size_t>(tOutputInfreq / dt_analytic + 0.5);
    }

    if (dt_analytic > t_analytic) {
        std::cerr << "dt_analytic should not larger than t_analytic: " << std::endl;
        std::exit(1);
    } else {
        Step_inteval = static_cast<size_t>(t_analytic / dt_analytic + 0.5);// static_cast<int>(t_analytic / dt_analytic + 0.5);
        // std::cout << "Step_inteval=" << t_analytic / dt_analytic << " " << t_analytic / dt_analytic + 0.5 << " " << Step_inteval << std::endl;
    }




    size_t OutputFreq_interval = static_cast<int>(tOutputFreq / dt + 0.5);
    size_t OutputInfreq_interval = static_cast<int>(tOutputInfreq / dt + 0.5);
    size_t Snap_interval = static_cast<int>(tSnapFreq / dt + 0.5);


    if (dt > tOutputFreq) {
        std::cerr << "dt should not larger than tOutputFreq: " << std::endl;
        std::exit(1);
    } else {
        OutputFreq_interval = static_cast<size_t>(tOutputFreq / dt + 0.5);
    }

    if (dt > tOutputInfreq) {
        std::cerr << "dt should not larger than tOutputInfreq: " << std::endl;
        std::exit(1);
    } else {
        OutputInfreq_interval = static_cast<size_t>(tOutputInfreq / dt + 0.5);
    }

    if (dt > tSnapFreq) {
        std::cerr << "dt should not larger than tSnapFreq: " << std::endl;
        std::exit(1);
    } else {
        Snap_interval = static_cast<size_t>(tSnapFreq / dt + 0.5);
    }


    // std::cout << "OutputFreq_interval = " << tOutputFreq / dt << " " << static_cast<int>(tOutputFreq / dt) << " "<< static_cast<int>(tOutputFreq / dt + 0.5) << std::endl;
    // std::cout << 



    double next_Output_Freq_time = tOutputFreq;
    double next_Output_Infreq_time = tOutputInfreq;


    #if ANALYTIC_EVOLUTION
        k_square_initialize();
        initialize_analytic_perturbation();

        #if BIFURCATION
        spectra_trajectory_analytic();
        #endif



        while(t < t_analytic) { // Main time evolution loop
            dt_analytic = calculate_dt(kIR * N * sqrt(3) / 2., a_k);

            // std::cout << "dt_analytic = " << dt_analytic << std::endl;
            dt_analytic = std::min(dt_analytic, tOutputFreq);

            if (t + dt_analytic > next_Output_Freq_time || t + dt_analytic > next_Output_Infreq_time) {
                dt_analytic = std::min(next_Output_Freq_time - t, next_Output_Infreq_time - t);
            }

        // while(numsteps < Step_inteval) { // Main time evolution loop
            // std::cout << "step=" << numsteps << std::endl;
            evolve_VV_analytic_perturbation(dt_analytic);
            numsteps++;


            if (std::abs(t - next_Output_Freq_time) < 1e-10) {
                next_Output_Freq_time += tOutputFreq;
            // if (numsteps % OutputFreq_interval_analytic == 0) {
                save_Freq_analytic();
            }

            if (std::abs(t - next_Output_Infreq_time) < 1e-10) {
                next_Output_Infreq_time += tOutputInfreq;
            // if (numsteps % OutputInfreq_interval_analytic == 0) {
                save_Infreq_analytic();
                #if BIFURCATION
                spectra_trajectory_analytic();
                #endif


                if (checkAllVectors(fk_left, pi_k_left, fk_right, pi_k_right)) {
                    std::cerr << "The program has detected a NaN value. Exiting." << std::endl;
                    return 1;
                }

                auto now = std::chrono::high_resolution_clock::now();
                elapsed = now - start;
                double total_seconds = elapsed.count();

                size_t hours = total_seconds / 3600;
                size_t minutes = (total_seconds - 3600 * hours) / 60;
                double seconds = total_seconds - 3600 * hours - 60 * minutes;

                std::cout << std::endl << "(";
                if (hours > 0) {
                    std::cout << hours << "h ";
                }
                if (minutes > 0 || hours > 0) { // Show minutes if there are hours
                    std::cout << minutes << "m ";
                }
                std::cout << seconds << "s)" << std::endl;



                // for (size_t fld = 0; fld < nflds; fld++) {
                //     for (size_t i = 0; i < num_k_values; ++i) {
                //         std::cout << "f_k[" << fld << "]["<< i << "]=" << fk_left[fld][i] << " ";
                //     }
                //     std::cout << std::endl << std::endl;
                // }


                std::cout << "Analytic Step " << numsteps << " done. Current time: " << t << std::endl;
                // output_ << t << std::endl;

                // std::cin.get();
            }
            
        }

        initialize_lattice();
            // #if BIFURCATION
            // spectra_trajectory_lattice();
            // // save_Infreq();
            // return(0);
            // #endif
    #else
        initialize();

        DECLARE_INDICES

        int px, py, pz;


        double f_av = 0.;
        double f2_av = 0.;
        double pi_av = 0.;
        double pi2_av = 0.;
        LOOP {
            f_av += f[0][i][j][k];
            f2_av += pw2(f[0][i][j][k]);
            pi_av += pi_f[0][i][j][k];
            pi2_av += pw2(pi_f[0][i][j][k]);
        }
        f_av /= N3;
        f2_av /= N3;
        pi_av /= N3;
        pi2_av /= N3;
        std::cout << "f_av = " << f_av << ", f2_av = " << f2_av << ", pi_av = " << pi_av << ", pi2_av = " << pi2_av << ", pi2_av/2 = " << pi2_av/2 << std::endl;


        #if BIFURCATION
        initialize_k();
        spectra_trajectory_lattice();
        #endif

        #if WITH_GW
        initialize_GW();
        // for (size_t i_GW = 0; i_GW < 3; i_GW++) {
        //     for (size_t j_GW = 0; j_GW < 3; j_GW++) {
        //         for (size_t i = 0; i < N; i++) {
        //             for (size_t j = 0; j < N; j++) {
        //                 for (size_t k = 0; k < N; k++) {
        //                     u[i_GW][j_GW][i][j][k] = 0.;
        //                     pi_u[i_GW][j_GW][i][j][k] = 0.;
        //                 }
        //                 for (size_t k = 0; k <= N/2; k++) {
        //                     u_k[i_GW][j_GW][(i * N + j) * (N/2+1) + k][0] = 0.;
        //                     u_k[i_GW][j_GW][(i * N + j) * (N/2+1) + k][1] = 0.;
        //                     pi_u_k[i_GW][j_GW][(i * N + j) * (N/2+1) + k][0] = 0.;
        //                     pi_u_k[i_GW][j_GW][(i * N + j) * (N/2+1) + k][1] = 0.;
        //                 }
        //             }
        //         }
        //     }
        // }

        #endif
        save();
    #endif




    #if BIFURCATION
    double a_buffer = a;
    double ad_buffer = ad;
    double add_buffer = add;
    double t_buffer = t;



    for (size_t fld = 0; fld < nflds; fld++) {
        f_k_initial[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));
        pi_k_initial[fld] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * (N/2 + 1));

        std::memcpy(f_initial[fld], f[fld], sizeof(double) * N * N * N);
        std::memcpy(pi_initial[fld], pi_f[fld], sizeof(double) * N * N * N);
        std::memcpy(f_k_initial[fld], f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
        std::memcpy(pi_k_initial[fld], pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
    }

        #if ANALYTIC_EVOLUTION
        for (size_t k = 1; k < maxnumbins; k++) {
            auto it = k_square.find(pw2(k));
            if (it == k_square.end()) {
                throw std::runtime_error("Element not found");
            }
            size_t position = std::distance(k_square.begin(), it);
            for (size_t fld = 0; fld < nflds; fld++) {
                fk_positive[fld][k] = fk_left[fld][position];
                pi_k_positive[fld][k] = pi_k_left[fld][position];
                
                fk_negative[fld][k] = fk_left[fld][position];
                pi_k_negative[fld][k] = pi_k_left[fld][position];
            }
        }
        #endif

    std::cout << std::endl << "Figuring out bifurcation... " << std::endl;

    while(t < tcheck) { // Main time evolution loop

        dt = calculate_dt(kIR * N * sqrt(3) / 2., a);
        ///////////////////////////  meijia you wenti /////////////////////////////
        dt = std::min(dt, tOutputFreq);
        ///////////////////////////  meijia you wenti /////////////////////////////

        if (t + dt > tcheck) {
            dt = tcheck - t;
        }

        #if SPECTRAL_FLAG
            #if PAD_FLAG
                evolve_VV_k_pad(dt);
            #else
                evolve_VV_k_nopad(dt);
            #endif
        #else
            evolve_VV(dt);
        #endif

        // std::cout << "t = " << t << ", dt = " << dt << std::endl;
    }

    bifurcation_sign();

    for (size_t fld = 0; fld < nflds; fld++) {
        std::memcpy(f[fld], f_initial[fld], sizeof(double) * N * N * N);
        std::memcpy(pi_f[fld], pi_initial[fld], sizeof(double) * N * N * N);
        std::memcpy(f_k[fld], f_k_initial[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
        std::memcpy(pi_k[fld], pi_k_initial[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
    }

    t = t_buffer;
    a = a_buffer;
    ad = ad_buffer;
    add = add_buffer;

    // std::cout << "t = " << t << ", a = " << a << ", ad = " << ad << ", add = " << add << std::endl;

    // std::cin.get();

    #endif
    
    
    // dt = calculate_dt(kIR * N * sqrt(3) / 2., a);


    // if (ssnapshot) {
    double next_Snap_time = t_analytic + tSnapFreq;
    // }

    numsteps = 0;

    size_t Lattice_Step_Inteval;
    if (dt > tf) {
        std::cerr << "dt should not larger than tf: " << std::endl;
        std::exit(1);
    } else {
        #if ANALYTIC_EVOLUTION
            Lattice_Step_Inteval = static_cast<size_t>((tf - t_analytic) / dt + 0.5);// static_cast<int>(t_analytic / dt_analytic + 0.5);
        // std::cout << "Step_inteval=" << t_analytic / dt_analytic << " " << t_analytic / dt_analytic + 0.5 << " " << Step_inteval << std::endl;
        #else
            Lattice_Step_Inteval = static_cast<size_t>(tf / dt + 0.5);
        #endif
    }

    while(t < tf) { // Main time evolution loop
    // while(numsteps < Lattice_Step_Inteval) {

        dt = calculate_dt(kIR * N * sqrt(3) / 2., a);
        dt = std::min(dt, tOutputFreq);

        if (t + dt > next_Output_Freq_time || t + dt > next_Output_Infreq_time) {
            dt = std::min(next_Output_Freq_time - t, next_Output_Infreq_time - t);
        }

        #if SPECTRAL_FLAG
            #if PAD_FLAG
                evolve_VV_k_pad(dt);
            #else
                #if BIFURCATION
                    evolve_VV_k_nopad_with_perturbation(dt);
                #else
                    evolve_VV_k_nopad(dt);
                #endif
            #endif
        #else

            #if BIFURCATION
                evolve_VV_with_perturbation(dt);
            #else
                evolve_VV(dt);
            #endif
        #endif

        // numsteps++;

        // std::cout << "numsteps / OutputFreq_interval" << numsteps % OutputFreq_interval << std::endl;

        if (std::abs(t - next_Output_Freq_time) < 1e-10) {
            next_Output_Freq_time += tOutputFreq;
        // if (numsteps % OutputFreq_interval == 0) {
            save_Freq();
            #if BIFURCATION
            meansvars_bifurcation();
            #endif
        }


        if (std::abs(t - next_Output_Infreq_time) < 1e-10) {
            next_Output_Infreq_time += tOutputInfreq;
        // if (numsteps % OutputInfreq_interval == 0) {
            save_Infreq();
            #if BIFURCATION
            spectra_trajectory_lattice();
            d2Vdf2_evolution();
            #endif

            auto now = std::chrono::high_resolution_clock::now();
            elapsed = now - start;
            double total_seconds = elapsed.count();

            size_t hours = total_seconds / 3600;
            size_t minutes = (total_seconds - 3600 * hours) / 60;
            double seconds = total_seconds - 3600 * hours - 60 * minutes;

            std::cout << std::endl << "(";
            if (hours > 0) {
                std::cout << hours << "h ";
            }
            if (minutes > 0 || hours > 0) { // Show minutes if there are hours
                std::cout << minutes << "m ";
            }
            std::cout << seconds << "s)" << std::endl;

            std::cout << "Lattice Step " << numsteps << " done. Current time: " << t << std::endl;
        }

        if (std::abs(t - next_Snap_time) < 1e-10) {
            next_Snap_time += tSnapFreq;
        // if (numsteps % Snap_interval == 0)
            numsteps++;
            save_Snap(numsteps);
        }
    }


    output_parameters();
    std::cout << "LATTICEEASY program finished" << std::endl;

    for (size_t fld = 0; fld < nflds; fld++) {
        fftw_free(f_k[fld]);
        fftw_free(pi_k[fld]);
        #if SPECTRAL_FLAG
            fftw_free(pi_k_half[fld]);
            fftw_free(V_prime_k[fld]);
            fftw_free(kernel_f_k[fld]);
            #if SPECTRAL_FLAG && PAD_FLAG
                fftw_free(f_k_pad[fld]);
                fftw_free(V_prime_k_pad[fld]);
            #endif
        #endif
    }

    #if WITH_GW
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 3; j++) {
                fftw_free(u_k[i][j]);
            }
    #endif
    fftw_destroy_plan(plan);
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_pi);
    #if SPECTRAL_FLAG && PAD_FLAG
        fftw_destroy_plan(plan_pad);
    #endif


    #if ANALYTIC_EVOLUTION
        std::cout << "num_k_values = " << num_k_values << std::endl;
    #endif


    #if BIFURCATION
        std::cout << "count_positive = " << count_positive << std::endl;
    #endif

    std::system("afplay /System/Library/Sounds/Glass.aiff");
    return(0);
}
