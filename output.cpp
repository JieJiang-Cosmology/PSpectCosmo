// Copyright (c) 2024 Jie Jiang (江捷) <jiejiang@pusan.ac.kr>
// This code is licensed under the MIT License.
// See the LICENSE file in the project root for license information.

#include "pspectcosmo.h"

std::string name_;

#if WITH_GW


// // 计算向量的模长
// double norm(const Vector3 &v) {
//     return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
// }

// Function to compute the projection operator P
Matrix3x3 projection_operator(const Vector3& k) {
    double k_norm = norm(k);
    if (k_norm == 0) {
        throw std::invalid_argument("动量向量 k 的模不能为零。");
    }

    Vector3 k_hat;
    for (int i = 0; i < 3; ++i) {
        k_hat[i] = k[i] / k_norm;
    }

    Matrix3x3 delta = { { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} } };
    Matrix3x3 P = {0};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            P[i][j] = delta[i][j] - k_hat[i] * k_hat[j];
        }
    }

    return P;
}


// TT projection
ComplexMatrix3x3 P_projection(const Matrix3x3& P, const ComplexMatrix3x3& e_TT) {
    ComplexMatrix3x3 v_ij = {0.};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                    v_ij[i][j] += P[i][k] * e_TT[k][j];
            }
        }
    }
    return v_ij;
}
#endif


std::vector<double> energy_density() {
    DECLARE_INDICES
    size_t fld;
    #if SPECTRAL_FLAG
    // if (isChanged & useSpectral) {
    if (isChanged) {
        fftw_plan forward_plan_f;
        fftw_plan forward_plan_pi;
        for (fld = 0; fld < nflds; fld++) {

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
            }
        }
        fftw_destroy_plan(forward_plan_f);
        fftw_destroy_plan(forward_plan_pi);
        isChanged = false;
    }
    #endif

    std::vector<double> energy_values;
    double total = 0.0;

    for (size_t fld = 0; fld < nflds; fld++) {
        #if !SPECTRAL_FLAG
            double deriv_energy = kinetic_energy(fld);
        #else
            double deriv_energy = kinetic_energy_k(fld);
        #endif
        total += deriv_energy;
        energy_values.push_back(deriv_energy);

        #if !SPECTRAL_FLAG
            double grad_energy = gradient_energy(fld);
        #else
            double grad_energy = gradient_energy_k(fld);
        #endif
        total += grad_energy;
        energy_values.push_back(grad_energy);
    }

    for (size_t i = 0; i < num_potential_terms; i++) {
        double pot_energy = potential_energy(i, nullptr);
        total += pot_energy;
        energy_values.push_back(pot_energy);
    }

    energy_values.push_back(total);
    return energy_values;
}




#if ANALYTIC_EVOLUTION
void meansvars_analytic() {
    static std::ofstream fields_[nflds];
    DECLARE_INDICES
    int fld;

    static bool isFirst = true;
    if (isFirst) {
        for (fld = 0; fld < nflds; fld++) {
            name_ = dir_ + "/average_scalar_" + std::to_string(fld) + ext_;
            if (continue_run)
                fields_[fld].open(name_, std::ios::out | std::ios::app);
            else
                fields_[fld].open(name_, std::ios::out | std::ios::trunc);


        }
        isFirst = false;
    }

    double a_of_pi = pow(a_k, alpha - 3);
    for (fld = 0; fld < nflds; fld++) {
        fields_[fld] << std::fixed << std::setprecision(Freq_prec) << t << std::scientific << std::setprecision(6) << std::setw(setw_leng) << fk_left[fld][0].real() << std::setw(setw_leng) << pi_k_left[fld][0].real() * a_of_pi << std::setw(setw_leng) << pw2(fk_left[fld][0].real()) << std::setw(15) << pw2(pi_k_left[fld][0].real() * a_of_pi) << std::setw(setw_leng) << 0 << std::setw(setw_leng) << 0 << std::endl;
    }
}

void scale_analytic() {
    static std::ofstream sf_;

    static bool isFirst = true;
    if (isFirst) {
        name_ = dir_ + "/average_scale_factor" + ext_;
        if (continue_run)
            sf_.open(name_, std::ios::out | std::ios::app);
        else
            sf_.open(name_, std::ios::out | std::ios::trunc);
        isFirst = false;
    }

    sf_ << std::fixed << std::setprecision(6) << t << " " << a_k << " " << ad_k << " " << ad_k / a_k << " " << add_k <<  " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << 1 / (pow(a_k, - alpha) * ad_k) <<  " " << std::fixed << std::setprecision(6) << 1 + alpha - a_k * add_k / pw2(ad_k) << std::endl;
    // sf_ << std::fixed << std::setprecision(6) << t << " " << a << " " << ad << " " << ad / a << " " << add <<  " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << 1 / (pow(a, - alpha) * ad) <<  " " << std::fixed << std::setprecision(6) << 1 + alpha - a * add / pw2(ad) << std::endl;
}


void spectra_analytic_discrete() {
    DECLARE_INDICES
    static std::ofstream spectra_[nflds], spectratimes_;

    static std::ofstream spectraEvolution_[nflds];

    size_t fld;

    double k_mode;
    double spectra[nflds];

    size_t position;

    static bool isFirst = true;
    if (isFirst) {
        for (fld = 0; fld < nflds; fld++) {
            name_ = dir_ + "/spectra_scalar_" + std::to_string(fld) + ext_;
            if (continue_run)
                spectra_[fld].open(name_, std::ios::out | std::ios::app);
            else
                spectra_[fld].open(name_, std::ios::out | std::ios::trunc);


            name_ = dir_ + "/spectra_scalar_" + std::to_string(fld) + "_evolution" + ext_;
            if (continue_run)
                spectraEvolution_[fld].open(name_, std::ios::out | std::ios::app);
            else
                spectraEvolution_[fld].open(name_, std::ios::out | std::ios::trunc);

            // spectraEvolution_[fld] << std::fixed << std::setprecision(kIR_prec) << 0 << " ";
            for (i = 0; i < maxnumbins; i++) {
                spectraEvolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " ;
                // spectraEvolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " ;
            }

            // for (i = 0; i < maxnumbins; i++) {
            //     // spectraEvolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " ;
            //     spectraEvolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " ;
            // }
            spectraEvolution_[fld] << std::endl;
        }

        name_ = dir_ + "/average_spectra_times" + ext_;
        if (continue_run)
            spectratimes_.open(name_, std::ios::out | std::ios::app);
        else
            spectratimes_.open(name_, std::ios::out | std::ios::trunc);


        isFirst = false;
    }

    // ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    double a_of_pi = pow(a_k, alpha - 3);
    // ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    for (fld = 0; fld < nflds; fld++) {
        spectraEvolution_[fld] << std::fixed << std::setprecision(6) << t << " ";
        for (k = 1; k < maxnumbins; k++) {

            auto it = k_square.find(pw2(k));
            position = std::distance(k_square.begin(), it);

            k_mode = sqrt(k_square_vec[position]) * kIR;
            // k_mode = sqrt(k_square_vec[position]) * kOverA;

            spectra[fld] = pw2(fk_left[fld][position].real()) + pw2(fk_left[fld][position].imag());

            spectra_[fld] << std::fixed << std::setprecision(kIR_prec) << k_mode << " " ;

            // to test
            // k_mode = sqrt(k_square_vec[position]) * kIR;
            
            spectra_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(k_mode, 3) / (2 * pw2(M_PI)) * spectra[fld];

            spectraEvolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << sqrt(spectra[fld]) << " ";
            // spectraEvolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << sqrt(spectra[fld]) << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << fk_left[fld][position].real() << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << fk_left[fld][position].imag() << " ";


            spectra[fld] = pw2(a_of_pi) * pow(k_mode, 3) / (2 * pw2(M_PI)) * (pw2(pi_k_left[fld][position].real()) + pw2(pi_k_left[fld][position].imag()));

            spectra_[fld] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << spectra[fld] << " " << 0 << " " << 0 << std::endl;
        }
        spectraEvolution_[fld] << std::endl;
    }


    for (fld = 0; fld < nflds; fld++) {
        spectra_[fld] << std::endl;
    }


    spectratimes_ << std::fixed << std::setprecision(Infreq_prec) << t << std::endl;
}



// void spectra_analytic() {
//     DECLARE_INDICES
//     static std::ofstream spectra_[nflds];
//     int fld;

//     double k_mode;
//     double spectra[nflds];

//     static bool isFirst = true;
//     if (isFirst) {
//         for (fld = 0; fld < nflds; fld++) {
//             name_ = dir_ + "/analytic_spectra_scalar_" + std::to_string(fld) + ext_;
//             if (continue_run)
//                 spectra_[fld].open(name_, std::ios::out | std::ios::app);
//             else
//                 spectra_[fld].open(name_, std::ios::out | std::ios::trunc);
//         }

//         isFirst = false;
//     }

//     // ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//     double a_of_pi = pow(a_k, alpha - 3);
//     // ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//     for (k = 1; k < num_k_values; k++) {
//         k_mode = sqrt(k_square_vec[k]) * kIR;

//         for (fld = 0; fld < nflds; fld++) {
//             spectra[fld] = pow(k_mode, 3) / (2 * pw2(M_PI)) * (pw2(fk_left[fld][k].real()) + pw2(fk_left[fld][k].imag()));

//             spectra_[fld] << std::fixed << std::setprecision(kIR_prec) << k_mode << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << spectra[fld];

//             spectra[fld] = pw2(a_of_pi) * pow(k_mode, 3) / (2 * pw2(M_PI)) * (pw2(pi_k_left[fld][k].real()) + pw2(pi_k_left[fld][k].imag()));

//             spectra_[fld] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << spectra[fld] << std::endl;
//         }
//     }


//     for (fld = 0; fld < nflds; fld++) {
//         spectra_[fld] << std::endl;
//     }
// }


void energy_analytic() {
    static std::ofstream energy_, conservation_;
    static double totalinitial; // Initial value of energy density (used for checking conservation in Minkowski space)
    DECLARE_INDICES
    int fld;
    double var, vard, ffd; // Averaged field values over the grid (defined below)
    double deriv_energy, grad_energy, pot_energy, total; // Total gives the total value of rho(t) for checking energy conservation
    double LHS, RHS;

    static bool isFirst = true;
    if (isFirst) { // Open output files (isFirst is set to zero at the bottom of the function)
        name_ = dir_ + "/average_energies" + ext_;
        // if (continue_run)
            energy_.open(name_, std::ios::out | std::ios::app);
        // else
        //     energy_.open(name_, std::ios::out | std::ios::trunc);

        if (expansion != 1) { // For power-law expansion don't calculate conservation
            name_ = dir_ + "/average_energy_conservation" + ext_;
            // if (continue_run)
                conservation_.open(name_, std::ios::out | std::ios::app);
            // else
            //     conservation_.open(name_, std::ios::out | std::ios::trunc);
        }
    } // The variable isFirst is used again at the end of the function, where it is then set to 0

    energy_ << std::fixed << std::setprecision(Freq_prec) << t;

    double field_point[nflds];

    total = 0.;
    // Calculate and output kinetic (time derivative) energy
    for (fld = 0; fld < nflds; fld++) { // Note that energy is output for all fields regardless of the value of nflds
        deriv_energy = pw2(pi_k_left[fld][0].real()) / pow(a_k, 6) / 2.;
        total += deriv_energy;
        energy_ << std::scientific << std::setprecision(6) << " " << deriv_energy;
    
        grad_energy = 0;
        total += grad_energy;
        energy_ << std::scientific << std::setprecision(6) << " " << grad_energy;
        field_point[fld] = fk_left[fld][0].real();
    }

    // Calculate and output potential energy
    for (i = 0; i < num_potential_terms; i++) {
        pot_energy = potential_energy(i, field_point); // Model dependent function for calculating potential energy terms
        total += pot_energy;
        energy_ << std::scientific << std::setprecision(6) << " " << pot_energy;
    }
    energy_ << std::scientific << std::setprecision(6) << " " << total << std::endl;

    // Energy conservation
    if (isFirst) { // In Minkowski space record the initial value of the energy to be used for checking energy conservation
        if (expansion == 0)
            totalinitial = total;
        isFirst = false; // Regardless of the expansion set isFirst to 0 so file streams aren't opened again.
    }

    if (expansion != 1) { // Conservation isn't checked for power law expansion
        if (expansion == 0) {// In Minkowski space the file conservation_ records the ratio of rho(t) to rho_initial
            conservation_ << std::fixed << std::setprecision(Freq_prec) << t << " " << std::scientific << std::setprecision(6) << total / totalinitial << std::endl;
        } else { // In an expanding universe the file conservation_ records the ratio of H^2(t) to 8 pi/3 rho(t)
            LHS = pw2(ad_k);
            RHS = pw2(fStar / MPl) * pow(a_k, 2 * (alpha + 1)) * total / 3.;
            conservation_ << std::fixed << std::setprecision(Freq_prec) << t << " " << std::scientific << std::setprecision(8) << fabs(LHS - RHS) / fabs(LHS + RHS) << " " << std::fixed << std::setprecision(6) << LHS << " " << RHS << std::endl;
        }
    }
}


#endif


void meansvars() {
    static std::ofstream fields_[nflds];
    DECLARE_INDICES
    int fld;
    double f_av[nflds], f_sq_av[nflds], var_f[nflds], var_fd[nflds];
    double pi_av[nflds], pi_sq_av[nflds];


    static bool isFirst = true;
    if (isFirst) {
        for (fld = 0; fld < nflds; fld++) {
            name_ = dir_ + "/average_scalar_" + std::to_string(fld) + ext_;
            // if (continue_run)
                fields_[fld].open(name_, std::ios::out | std::ios::app);
            // else
            //     fields_[fld].open(name_, std::ios::out | std::ios::trunc);
        }

        isFirst = false;
    }

    #if SPECTRAL_FLAG
    if (isChanged) {
        fftw_plan forward_plan_f;
        fftw_plan forward_plan_pi;
        for (fld = 0; fld < nflds; fld++) {

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
            }

        }
        fftw_destroy_plan(forward_plan_f);
        fftw_destroy_plan(forward_plan_pi);
        isChanged = false;
    }
    #endif

    double a_of_pi = pow(a, alpha - 3);
    for (fld = 0; fld < nflds; fld++) {
        f_av[fld]    = 0.;
        f_sq_av[fld] = 0.;

        pi_av[fld]   = 0.;
        pi_sq_av[fld] = 0.;


        LOOP {
            f_av[fld]  += FIELD(fld);
            f_sq_av[fld] += pw2(FIELD(fld));

            pi_av[fld] += PI_FIELD(fld);
            pi_sq_av[fld] += pw2(PI_FIELD(fld));
        }

        f_av[fld]     /= N3;
        f_sq_av[fld]  /= N3;

        pi_av[fld] /= N3;
        pi_sq_av[fld] /= N3;

        if (f_av[fld] + DBL_MAX == f_av[fld] || (f_av[fld] != 0. && f_av[fld] / f_av[fld] != 1.)) {
            std::cerr << "Unstable solution developed. Field " << fld << " not numerical at t = " << t << std::endl << "If you have used analytic evolution to evolve perturbations, please adjust your t_analytic to be smaller, and make sure the scale factor is greater than 1 when the analytic evolution stops." << std::endl;
            output_parameters();
            std::exit(1);
        }

        var_f[fld]  = (f_sq_av[fld] - pw2(f_av[fld])  > 0 ? sqrt(f_sq_av[fld] - pw2(f_av[fld]))   : 0);
        var_fd[fld] = (pi_sq_av[fld] - pw2(pi_av[fld]) > 0 ? a_of_pi * sqrt(pi_sq_av[fld] - pw2(pi_av[fld])) : 0);

        fields_[fld] << std::fixed << std::setprecision(Freq_prec) << t << std::scientific << std::setprecision(6) << std::setw(setw_leng) << f_av[fld] << std::setw(setw_leng) << a_of_pi * pi_av[fld] << std::setw(setw_leng) << f_sq_av[fld] << std::setw(15) << pw2(a_of_pi) * pi_sq_av[fld] << std::setw(setw_leng) << var_f[fld] << std::setw(setw_leng) << var_fd[fld] << std::endl;
    }
}





void scale() {
    static std::ofstream sf_;

    static bool isFirst = true;
    if (isFirst) {
        name_ = dir_ + "/average_scale_factor" + ext_;
        // if (continue_run)
            sf_.open(name_, std::ios::out | std::ios::app);
        // else
        //     sf_.open(name_, std::ios::out | std::ios::trunc);
        isFirst = false;
    }

    if (a < 0) {
        std::cerr << "Error encountered at t = "<< t << ", a is negative (" << a << "). Execution stopped. Please set dt to a smaller value to ensure convergence. " << std::endl;
        exit(1);  // 使用非零值退出，表示错误
    }

    sf_ << std::fixed << std::setprecision(6) << t << " " << a << " " << ad << " " << ad / a << " " << add <<  " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << 1 / (pow(a, - alpha) * ad) <<  " " << std::fixed << std::setprecision(6) << 1 + alpha - a * add / pw2(ad) << std::endl;
}



void spectra() {
    static std::ofstream spectra_[nflds], spectratimes_;
    // static std::ofstream curvature_spectra_;

    static std::ofstream spectraEvolution_[nflds];


    size_t fld;
    int numpoints[maxnumbins];
    double p[maxnumbins], f2[maxnumbins], pi_f2[maxnumbins], ndensity[maxnumbins], n_2[maxnumbins]; // Values for each bin: Momentum, |f_k|^2, |f_k'|^2, n_k
    size_t l;

    #if WITH_GW
    static std::ofstream spectra_GWs_, GWs_energy_;
    size_t i_GW, j_GW;
    #endif

    // double pi_f2[maxnumbins];
    double fp2, fp2_R, fp2_I, pi_fp2, np2;
    double omega, omegasq;
    double mass_sq[nflds], mass_sq_av[nflds], point_field[nflds];

    DECLARE_INDICES
    int px, py, pz;

    static bool isFirst = true;
    if (isFirst) {
        for (fld = 0; fld < nflds; fld++) {
            name_ = dir_ + "/spectra_scalar_" + std::to_string(fld) + ext_;
            // if (continue_run)
                spectra_[fld].open(name_, std::ios::out | std::ios::app);
            // else
            //     spectra_[fld].open(name_, std::ios::out | std::ios::trunc);


            name_ = dir_ + "/spectra_scalar_" + std::to_string(fld) + "_evolution" + ext_;
            spectraEvolution_[fld].open(name_, std::ios::out | std::ios::app);

            #if !ANALYTIC_EVOLUTION
                for (i = 0; i < maxnumbins; i++) {
                    spectraEvolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " ;
                    // spectraEvolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " ;
                    // spectraEvolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kOverA * i << " " ;
                }
                spectraEvolution_[fld] << std::endl;
            #endif
        }
        name_ = dir_ + "/average_spectra_times" + ext_;
        // if (continue_run)
            spectratimes_.open(name_, std::ios::out | std::ios::app);
        // else
        //     spectratimes_.open(name_, std::ios::out | std::ios::trunc);


        #if WITH_GW
        name_ = dir_ + "/spectra_gws" + ext_;
        // if (continue_run)
            spectra_GWs_.open(name_, std::ios::out | std::ios::app);
        // else
        //     spectra_[fld].open(name_, std::ios::out | std::ios::trunc);

        name_ = dir_ + "/energy_gws" + ext_;
        // if (continue_run)
            GWs_energy_.open(name_, std::ios::out | std::ios::app);
        // else
        //     spectratimes_.open(name_, std::ios::out | std::ios::trunc);
        #endif




        // name_ = dir_ + "/curvature_spectra" + ext_;
        // curvature_spectra_.open(name_, std::ios::out | std::ios::app);

        isFirst = false;
    }


    for (fld = 0; fld < nflds; fld++)
        mass_sq_av[fld] = 0.;

    LOOP {
        for (fld = 0; fld < nflds; fld++)
            point_field[fld] = f[fld][i][j][k];
        effective_mass(mass_sq, point_field);
        for (fld = 0; fld < nflds; fld++)
            mass_sq_av[fld] += mass_sq[fld];
    }

    for (fld = 0; fld < nflds; fld++)
        mass_sq_av[fld] /= N3;




    #if !SPECTRAL_FLAG
    // if (isChanged & !useSpectral) {
    if (isChanged) {

        fftw_plan forward_plan_f;
        fftw_plan forward_plan_fd;

        for (fld = 0; fld < nflds; fld++) {
            forward_plan_f = fftw_plan_dft_r2c_3d(N, N, N, &f[fld][0][0][0], f_k[fld], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            forward_plan_fd = fftw_plan_dft_r2c_3d(N, N, N, &pi_f[fld][0][0][0], pi_k[fld], FFTW_ESTIMATE);
            fftw_execute(forward_plan_fd);
        }


        #if WITH_GW
        for(i_GW = 0; i_GW < 3; i_GW++) {
            for(j_GW = 0; j_GW < 3; j_GW++) {

                forward_plan_f = fftw_plan_dft_r2c_3d(N, N, N, &u[i_GW][j_GW][0][0][0], u_k[i_GW][j_GW], FFTW_ESTIMATE);
                fftw_execute(forward_plan_f);

                forward_plan_fd = fftw_plan_dft_r2c_3d(N, N, N, &pi_u[i_GW][j_GW][0][0][0], pi_u_k[i_GW][j_GW], FFTW_ESTIMATE);
                fftw_execute(forward_plan_fd);
            }
        }
        #endif

        fftw_destroy_plan(forward_plan_f);
        fftw_destroy_plan(forward_plan_fd);
        isChanged = false;
    }
    #endif



    double a_of_pi = pow(a, alpha - 3);



    for (fld = 0; fld < nflds; fld++) {
        for (i = 0; i < maxnumbins; i++) {
            numpoints[i] = 0; // Number of points in the bin
            f2[i] = 0.; // |f_p|^2
            pi_f2[i] = 0.;
            n_2[i] = 0.;
            ndensity[i] = 0.;
        }


        for (i = 0; i < N; i++) {
            px = (i <= N/2 ? i : i - N);
            for (j = 0; j < N; j++) {
                py = (j <= N/2 ? j : j - N);
                for (k = 1; k < N/2; k++) {
                    pz = k;
                    l = std::floor(sqrt(pw2(px) + pw2(py) + pw2(pz))+0.5);

                    fp2 = pw2(f_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(f_k[fld][(i * N + j) * (N/2+1) + k][1]);

                    pi_fp2 = pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][1]);

                    np2 = pw2(a_of_pi * pi_k[fld][(i * N + j) * (N/2+1) + k][0] + ad / a * f_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(a_of_pi * pi_k[fld][(i * N + j) * (N/2+1) + k][1] + ad / a * f_k[fld][(i * N + j) * (N/2+1) + k][1]);

                    numpoints[l] += 2;
                    f2[l] += 2. * fp2;
                    pi_f2[l] += 2. * pi_fp2;
                    n_2[l] += 2. * np2;
                }
                for (k = 0; k <= N/2; k += N/2) {
                    pz = k;
                    l = std::floor(sqrt(pw2(px) + pw2(py) + pw2(pz))+0.5);
                    
                    fp2 = pw2(f_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(f_k[fld][(i * N + j) * (N/2+1) + k][1]);
                    
                    pi_fp2 = pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][1]);

                    np2 = pw2(a_of_pi * pi_k[fld][(i * N + j) * (N/2+1) + k][0] + ad / a * f_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(a_of_pi * pi_k[fld][(i * N + j) * (N/2+1) + k][1] + ad / a * f_k[fld][(i * N + j) * (N/2+1) + k][1]);

                    numpoints[l]++;
                    f2[l] += fp2;
                    pi_f2[l] += pi_fp2;
                    n_2[l] += np2;
                }
            }
        }


        for (i = 1; i < maxnumbins; i++) {
            p[i] = kIR * i;
            if (numpoints[i] > 0) {
                f2[i]  = f2[i] / numpoints[i];
                pi_f2[i] = pi_f2[i] / numpoints[i];
                n_2[i] = n_2[i] / numpoints[i];
            }
            omegasq = fabs(pw2(p[i]) + pw2(a) * mass_sq_av[fld]);
            if (omegasq > 0.) {
                omega = sqrt(omegasq);
                ndensity[i] = pw2(a) * pow(L, 3) / 2. / pw2(N3) * pw2(fStar / omegaStar) * (omega * f2[i] + pow(a, 2 * (1 - alpha)) / omega * n_2[i]);
            }

            spectra_[fld] << std::fixed << std::setprecision(kIR_prec) << p[i] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << i / pw2(N3) * numpoints[i] * f2[i] << " " << std::setw(setw_leng) << i / pw2(N3) * numpoints[i] * pw2(a_of_pi) * pi_f2[i] << " " << std::setw(setw_leng) << ndensity[i] << " " << numpoints[i] << std::endl;
            // spectra_[fld] << std::fixed << std::setprecision(kIR_prec) << p[i] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << i / pw2(N3) * numpoints[i] * f2[i] * pow(N/dx, 2) << " " << std::setw(setw_leng) << i / pw2(N3) * numpoints[i] * pw2(a_of_pi) * pi_f2[i] << " " << std::setw(setw_leng) << ndensity[i] << " " << numpoints[i] << std::endl;
            // spectra_[fld] << std::fixed << std::setprecision(kIR_prec) << p[i] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << i / pw2(N3) * numpoints[i] * f2[i] << " " << std::setw(setw_leng) << i / pw2(N3) * numpoints[i] * pw2(a_of_pi) * pi_f2[i] << " " << std::setw(setw_leng) << ndensity[i] << " " << numpoints[i] << std::endl;
        }

        spectra_[fld] << std::endl;


        spectraEvolution_[fld] << std::fixed << std::setprecision(6) << t << " ";
        for (i = 1; i < maxnumbins; i++) {
            p[i] = kIR * i;

            // spectraEvolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(L, 1.5) / N3 * sqrt(f2[i]) << " ";
            // spectraEvolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << (N / dx) * pow(L, 1.5) / N3 * sqrt(f2[i]) << " ";
            spectraEvolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << (N / dx) * omegaStar * pow(L, 1.5) / N3 * sqrt(f2[i]) << " ";
        }
        spectraEvolution_[fld] << std::endl;

    }

    spectratimes_ << std::fixed << std::setprecision(Infreq_prec) << t << std::endl;

    
    





    // LOOP {
    //     R[i][j][k] = 0.;
    //     for (fld = 0; fld < nflds; fld++) {
    //         R[i][j][k] += pi_f[fld][i][j][k] * f[fld][i][j][k] * a_of_pi;
    //     }
    //     double R_denominator = 0.;
    //     for (fld = 0; fld < nflds; fld++) {
    //         R_denominator += pw2(pi_f[fld][i][j][k] * a_of_pi);
    //     }
    //     R[i][j][k] /= R_denominator;
    // }


    // fftw_plan plan_R = fftw_plan_dft_r2c_3d(N, N, N, &R[0][0][0], R_k, FFTW_ESTIMATE);
    // fftw_execute(plan_R);
    // fftw_destroy_plan(plan_R);

    // double R2[maxnumbins];
    // double Rp2;

    // for (i = 0; i < maxnumbins; i++) {
    //     R2[i] = 0.;
    // }

    // for (i = 0; i < N; i++) {
    //     px = (i <= N/2 ? i : i - N);
    //     for (j = 0; j < N; j++) {
    //         py = (j <= N/2 ? j : j - N);
    //         for (k = 1; k < N/2; k++) {
    //             pz = k;
    //             l = std::floor(sqrt(pw2(px) + pw2(py) + pw2(pz))+0.5);

    //             // numpoints[l] += 2;

    //             Rp2 = pw2(R_k[(i * N + j) * (N/2+1) + k][0]) + pw2(R_k[(i * N + j) * (N/2+1) + k][1]);
    //             // std::cout << "Rp2=" << Rp2 << std::endl;
    //             R2[l] += 2. * Rp2;
    //         }
    //         for (k = 0; k <= N/2; k += N/2) {
    //             pz = k;
    //             l = std::floor(sqrt(pw2(px) + pw2(py) + pw2(pz))+0.5);

    //             // numpoints[l]++;

    //             Rp2 = pw2(R_k[(i * N + j) * (N/2+1) + k][0]) + pw2(R_k[(i * N + j) * (N/2+1) + k][1]);
    //             R2[l] += Rp2;
    //         }
    //     }
    // }

    // for (i = 1; i < maxnumbins; i++) {
    //     p[i] = kIR * i;
    //     if (numpoints[i] > 0) {
    //         R2[i] = R2[i] / numpoints[i];
    //     }
    //     // std::cout << "R2[" << i << "] = " << R2[i] << std::endl;
    //     // std::cin.get();
    //     curvature_spectra_ << std::fixed << std::setprecision(kIR_prec) << p[i] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << i / pw2(N3) * numpoints[i] * pw2(ad / a) * R2[i] << " " << numpoints[i] << std::endl;
    // }
    // curvature_spectra_ << std::endl;




    #if WITH_GW
    // GWs spectrum
    ComplexMatrix3x3 h_ij, u_ij, hdot_ij, pi_u_ij;
    ComplexMatrix3x3 v_ij, vdot_ij;
    Vector3 k_vec;

    double h2[maxnumbins], hdot2[maxnumbins];


    for (i = 0; i < maxnumbins; i++) {
        numpoints[i] = 0; // Number of points in the bin
        h2[i] = 0.; // |f_p|^2
        hdot2[i] = 0.;
    }


    size_t k_index;
    for (k = 0; k <= N/2; k += N/2) { // On the k=0, N/2 plane
        for (i = 0; i <= N/2; i += N/2) {
            for (j = 1; j < N/2; j++) {
                k_vec = {i * kIR, j * kIR, k * kIR}; // (+x, +y) mode
                // k_vec = {i * kOverA, j * kOverA, k * kOverA}; // (+x, +y) mode

                // auto Lambda = Lambda_operator(k_vec);
                Matrix3x3 P = projection_operator(k_vec);

                k_index = (i * N + j) * (N/2+1) + k;
                for (i_GW = 0; i_GW < 3; i_GW++) {
                    for (j_GW = 0; j_GW < 3; j_GW++) {
                        u_ij[i_GW][j_GW].real(u_k[i_GW][j_GW][k_index][0]);
                        u_ij[i_GW][j_GW].imag(u_k[i_GW][j_GW][k_index][1]);

                        pi_u_ij[i_GW][j_GW].real(pi_u_k[i_GW][j_GW][k_index][0]);
                        pi_u_ij[i_GW][j_GW].imag(pi_u_k[i_GW][j_GW][k_index][1]);
                    }
                }

                // h_ij = TT_projection(Lambda, u_ij);
                // hdot_ij = TT_projection(Lambda, pi_u_ij);
                v_ij = P_projection(P, u_ij);
                vdot_ij = P_projection(P, pi_u_ij);

                l = std::floor(sqrt(pw2(i) + pw2(j) + pw2(k))+0.5);
                numpoints[l] += 2;
                // for (i_GW = 0; i_GW < 3; i_GW++) {
                //     for (j_GW = 0; j_GW < 3; j_GW++) {
                //         h2[l] += 2 * (pw2(h_ij[i_GW][j_GW].real()) + pw2(h_ij[i_GW][j_GW].imag()));
                //         hdot2[l] += 2 * (pw2(hdot_ij[i_GW][j_GW].real()) + pw2(hdot_ij[i_GW][j_GW].imag()));
                //     }
                // }
                h2[l] += 2 * (norm(v_ij[0][0]) + norm(v_ij[1][1]) + norm(v_ij[2][2]) + norm(v_ij[0][1]) + norm(v_ij[1][0]) + norm(v_ij[0][2]) + norm(v_ij[2][0]) + norm(v_ij[1][2]) + norm(v_ij[2][1]) - norm(v_ij[0][0] + v_ij[1][1] + v_ij[2][2]) / 2.);
                hdot2[l] += 2 * (norm(vdot_ij[0][0]) + norm(vdot_ij[1][1]) + norm(vdot_ij[2][2]) + norm(vdot_ij[0][1]) + norm(vdot_ij[1][0]) + norm(vdot_ij[0][2]) + norm(vdot_ij[2][0]) + norm(vdot_ij[1][2]) + norm(vdot_ij[2][1]) - norm(vdot_ij[0][0] + vdot_ij[1][1] + vdot_ij[2][2]) / 2.);
            }
        }
        for (j = 0; j <= N/2; j += N/2) {
            for (i = 1; i < N/2; i++) {
                k_vec = {i * kIR, j * kIR, k * kIR}; // (+x, +y) mode
                // k_vec = {i * kOverA, j * kOverA, k * kOverA}; // (+x, +y) mode

                // auto Lambda = Lambda_operator(k_vec);
                Matrix3x3 P = projection_operator(k_vec);

                k_index = (i * N + j) * (N/2+1) + k;
                for (i_GW = 0; i_GW < 3; i_GW++) {
                    for (j_GW = 0; j_GW < 3; j_GW++) {
                        u_ij[i_GW][j_GW].real(u_k[i_GW][j_GW][k_index][0]);
                        u_ij[i_GW][j_GW].imag(u_k[i_GW][j_GW][k_index][1]);

                        pi_u_ij[i_GW][j_GW].real(pi_u_k[i_GW][j_GW][k_index][0]);
                        pi_u_ij[i_GW][j_GW].imag(pi_u_k[i_GW][j_GW][k_index][1]);
                    }
                }

                // h_ij = TT_projection(Lambda, u_ij);
                // hdot_ij = TT_projection(Lambda, pi_u_ij);
                v_ij = P_projection(P, u_ij);
                vdot_ij = P_projection(P, pi_u_ij);

                l = std::floor(sqrt(pw2(i) + pw2(j) + pw2(k))+0.5);
                numpoints[l] += 2;
                // for (i_GW = 0; i_GW < 3; i_GW++) {
                //     for (j_GW = 0; j_GW < 3; j_GW++) {
                //         h2[l] += 2 * (pw2(h_ij[i_GW][j_GW].real()) + pw2(h_ij[i_GW][j_GW].imag()));
                //         hdot2[l] += 2 * (pw2(hdot_ij[i_GW][j_GW].real()) + pw2(hdot_ij[i_GW][j_GW].imag()));
                //     }
                // }
                h2[l] += 2 * (norm(v_ij[0][0]) + norm(v_ij[1][1]) + norm(v_ij[2][2]) + norm(v_ij[0][1]) + norm(v_ij[1][0]) + norm(v_ij[0][2]) + norm(v_ij[2][0]) + norm(v_ij[1][2]) + norm(v_ij[2][1]) - norm(v_ij[0][0] + v_ij[1][1] + v_ij[2][2]) / 2.);
                hdot2[l] += 2 * (norm(vdot_ij[0][0]) + norm(vdot_ij[1][1]) + norm(vdot_ij[2][2]) + norm(vdot_ij[0][1]) + norm(vdot_ij[1][0]) + norm(vdot_ij[0][2]) + norm(vdot_ij[2][0]) + norm(vdot_ij[1][2]) + norm(vdot_ij[2][1]) - norm(vdot_ij[0][0] + vdot_ij[1][1] + vdot_ij[2][2]) / 2.);
            }
        }
        for (i = 1; i < N/2; i++) {
            for (j = 1; j < N/2; j++) {
                k_vec = {i * kIR, j * kIR, k * kIR}; // (+x, +y) mode
                // k_vec = {i * kOverA, j * kOverA, k * kOverA}; // (+x, +y) mode

                // auto Lambda = Lambda_operator(k_vec);
                Matrix3x3 P = projection_operator(k_vec);

                k_index = (i * N + j) * (N/2+1) + k;
                for (i_GW = 0; i_GW < 3; i_GW++) {
                    for (j_GW = 0; j_GW < 3; j_GW++) {
                        u_ij[i_GW][j_GW].real(u_k[i_GW][j_GW][k_index][0]);
                        u_ij[i_GW][j_GW].imag(u_k[i_GW][j_GW][k_index][1]);

                        pi_u_ij[i_GW][j_GW].real(pi_u_k[i_GW][j_GW][k_index][0]);
                        pi_u_ij[i_GW][j_GW].imag(pi_u_k[i_GW][j_GW][k_index][1]);
                    }
                }

                // h_ij = TT_projection(Lambda, u_ij);
                // hdot_ij = TT_projection(Lambda, pi_u_ij);
                v_ij = P_projection(P, u_ij);
                vdot_ij = P_projection(P, pi_u_ij);

                l = std::floor(sqrt(pw2(i) + pw2(j) + pw2(k))+0.5);
                numpoints[l] += 2;
                // for (i_GW = 0; i_GW < 3; i_GW++) {
                //     for (j_GW = 0; j_GW < 3; j_GW++) {
                //         h2[l] += 2 * (pw2(h_ij[i_GW][j_GW].real()) + pw2(h_ij[i_GW][j_GW].imag()));
                //         hdot2[l] += 2 * (pw2(hdot_ij[i_GW][j_GW].real()) + pw2(hdot_ij[i_GW][j_GW].imag()));
                //     }
                // }
                h2[l] += 2 * (norm(v_ij[0][0]) + norm(v_ij[1][1]) + norm(v_ij[2][2]) + norm(v_ij[0][1]) + norm(v_ij[1][0]) + norm(v_ij[0][2]) + norm(v_ij[2][0]) + norm(v_ij[1][2]) + norm(v_ij[2][1]) - norm(v_ij[0][0] + v_ij[1][1] + v_ij[2][2]) / 2.);
                hdot2[l] += 2 * (norm(vdot_ij[0][0]) + norm(vdot_ij[1][1]) + norm(vdot_ij[2][2]) + norm(vdot_ij[0][1]) + norm(vdot_ij[1][0]) + norm(vdot_ij[0][2]) + norm(vdot_ij[2][0]) + norm(vdot_ij[1][2]) + norm(vdot_ij[2][1]) - norm(vdot_ij[0][0] + vdot_ij[1][1] + vdot_ij[2][2]) / 2.);
            }
            for (j = N/2+1; j < N; j++) {
                k_vec = {i * kIR, (j - N) * kIR, k * kIR}; // (+x, -y) mode
                // k_vec = {i * kOverA, (j - N) * kOverA, k * kOverA}; // (+x, -y) mode

                // auto Lambda = Lambda_operator(k_vec);
                Matrix3x3 P = projection_operator(k_vec);

                k_index = (i * N + j) * (N/2+1) + k;
                for (i_GW = 0; i_GW < 3; i_GW++) {
                    for (j_GW = 0; j_GW < 3; j_GW++) {
                        u_ij[i_GW][j_GW].real(u_k[i_GW][j_GW][k_index][0]);
                        u_ij[i_GW][j_GW].imag(u_k[i_GW][j_GW][k_index][1]);

                        pi_u_ij[i_GW][j_GW].real(pi_u_k[i_GW][j_GW][k_index][0]);
                        pi_u_ij[i_GW][j_GW].imag(pi_u_k[i_GW][j_GW][k_index][1]);
                    }
                }

                // h_ij = TT_projection(Lambda, u_ij);
                // hdot_ij = TT_projection(Lambda, pi_u_ij);
                v_ij = P_projection(P, u_ij);
                vdot_ij = P_projection(P, pi_u_ij);

                l = std::floor(sqrt(pw2(i) + pw2(j - N) + pw2(k))+0.5);
                numpoints[l] += 2;
                // for (i_GW = 0; i_GW < 3; i_GW++) {
                //     for (j_GW = 0; j_GW < 3; j_GW++) {
                //         h2[l] += 2 * (pw2(h_ij[i_GW][j_GW].real()) + pw2(h_ij[i_GW][j_GW].imag()));
                //         hdot2[l] += 2 * (pw2(hdot_ij[i_GW][j_GW].real()) + pw2(hdot_ij[i_GW][j_GW].imag()));
                //     }
                // }
                h2[l] += 2 * (norm(v_ij[0][0]) + norm(v_ij[1][1]) + norm(v_ij[2][2]) + norm(v_ij[0][1]) + norm(v_ij[1][0]) + norm(v_ij[0][2]) + norm(v_ij[2][0]) + norm(v_ij[1][2]) + norm(v_ij[2][1]) - norm(v_ij[0][0] + v_ij[1][1] + v_ij[2][2]) / 2.);
                hdot2[l] += 2 * (norm(vdot_ij[0][0]) + norm(vdot_ij[1][1]) + norm(vdot_ij[2][2]) + norm(vdot_ij[0][1]) + norm(vdot_ij[1][0]) + norm(vdot_ij[0][2]) + norm(vdot_ij[2][0]) + norm(vdot_ij[1][2]) + norm(vdot_ij[2][1]) - norm(vdot_ij[0][0] + vdot_ij[1][1] + vdot_ij[2][2]) / 2.);
            }
        }
    }

    for (k = 1; k < N/2; k++) { // except the k=0, N/2 plane, include the implicit conjugate mode
        for (i = 0; i < N; i++) {
            px = (i <= N/2 ? i : i - N);
            for (j = 0; j < N; j++) {
                py = (j <= N/2 ? j : j - N);

                k_vec = {px * kIR, py * kIR, k * kIR}; // (+x, +y) mode
                // k_vec = {px * kOverA, py * kOverA, k * kOverA}; // (+x, +y) mode

                // auto Lambda = Lambda_operator(k_vec);
                Matrix3x3 P = projection_operator(k_vec);

                k_index = (i * N + j) * (N/2+1) + k;
                for (i_GW = 0; i_GW < 3; i_GW++) {
                    for (j_GW = 0; j_GW < 3; j_GW++) {
                        u_ij[i_GW][j_GW].real(u_k[i_GW][j_GW][k_index][0]);
                        u_ij[i_GW][j_GW].imag(u_k[i_GW][j_GW][k_index][1]);

                        pi_u_ij[i_GW][j_GW].real(pi_u_k[i_GW][j_GW][k_index][0]);
                        pi_u_ij[i_GW][j_GW].imag(pi_u_k[i_GW][j_GW][k_index][1]);
                    }
                }

                // h_ij = TT_projection(Lambda, u_ij);
                // hdot_ij = TT_projection(Lambda, pi_u_ij);
                v_ij = P_projection(P, u_ij);
                vdot_ij = P_projection(P, pi_u_ij);

                l = std::floor(sqrt(pw2(px) + pw2(py) + pw2(k))+0.5);
                numpoints[l] += 2;
                // for (i_GW = 0; i_GW < 3; i_GW++) {
                //     for (j_GW = 0; j_GW < 3; j_GW++) {
                //         h2[l] += 2 * (pw2(h_ij[i_GW][j_GW].real()) + pw2(h_ij[i_GW][j_GW].imag()));
                //         hdot2[l] += 2 * (pw2(hdot_ij[i_GW][j_GW].real()) + pw2(hdot_ij[i_GW][j_GW].imag()));
                //     }
                // }
                h2[l] += 2 * (norm(v_ij[0][0]) + norm(v_ij[1][1]) + norm(v_ij[2][2]) + norm(v_ij[0][1]) + norm(v_ij[1][0]) + norm(v_ij[0][2]) + norm(v_ij[2][0]) + norm(v_ij[1][2]) + norm(v_ij[2][1]) - norm(v_ij[0][0] + v_ij[1][1] + v_ij[2][2]) / 2.);
                hdot2[l] += 2 * (norm(vdot_ij[0][0]) + norm(vdot_ij[1][1]) + norm(vdot_ij[2][2]) + norm(vdot_ij[0][1]) + norm(vdot_ij[1][0]) + norm(vdot_ij[0][2]) + norm(vdot_ij[2][0]) + norm(vdot_ij[1][2]) + norm(vdot_ij[2][1]) - norm(vdot_ij[0][0] + vdot_ij[1][1] + vdot_ij[2][2]) / 2.);
            }
        }
    }

    for (k = 0; k <= N/2; k += N/2) { // set the 8 "corners", to real
        for (i = 0; i <= N/2; i += N/2) {
            for (j = 0; j <= N/2; j += N/2) {
                if (i == 0 && j == 0 && k == 0) {
                    continue; // Skip the case where i = 0, j = 0, and k = 0
                }

                k_vec = {i * kIR, j * kIR, k * kIR}; // (+x, +y) mode
                // k_vec = {i * kOverA, j * kOverA, k * kOverA}; // (+x, +y) mode

                // auto Lambda = Lambda_operator(k_vec);
                Matrix3x3 P = projection_operator(k_vec);

                k_index = (i * N + j) * (N/2+1) + k;
                for (i_GW = 0; i_GW < 3; i_GW++) {
                    for (j_GW = 0; j_GW < 3; j_GW++) {
                        u_ij[i_GW][j_GW].real(u_k[i_GW][j_GW][k_index][0]);
                        u_ij[i_GW][j_GW].imag(u_k[i_GW][j_GW][k_index][1]);

                        pi_u_ij[i_GW][j_GW].real(pi_u_k[i_GW][j_GW][k_index][0]);
                        pi_u_ij[i_GW][j_GW].imag(pi_u_k[i_GW][j_GW][k_index][1]);
                    }
                }

                // h_ij = TT_projection(Lambda, u_ij);
                // hdot_ij = TT_projection(Lambda, pi_u_ij);
                v_ij = P_projection(P, u_ij);
                vdot_ij = P_projection(P, pi_u_ij);

                l = std::floor(sqrt(pw2(i) + pw2(j) + pw2(k))+0.5);
                numpoints[l] ++;
                // for (i_GW = 0; i_GW < 3; i_GW++) {
                //     for (j_GW = 0; j_GW < 3; j_GW++) {
                //         h2[l] += pw2(h_ij[i_GW][j_GW].real()) + pw2(h_ij[i_GW][j_GW].imag());
                //         hdot2[l] += pw2(hdot_ij[i_GW][j_GW].real()) + pw2(hdot_ij[i_GW][j_GW].imag());
                //     }
                // }
                h2[l] += norm(v_ij[0][0]) + norm(v_ij[1][1]) + norm(v_ij[2][2]) + norm(v_ij[0][1]) + norm(v_ij[1][0]) + norm(v_ij[0][2]) + norm(v_ij[2][0]) + norm(v_ij[1][2]) + norm(v_ij[2][1]) - norm(v_ij[0][0] + v_ij[1][1] + v_ij[2][2]) / 2.;
                hdot2[l] += norm(vdot_ij[0][0]) + norm(vdot_ij[1][1]) + norm(vdot_ij[2][2]) + norm(vdot_ij[0][1]) + norm(vdot_ij[1][0]) + norm(vdot_ij[0][2]) + norm(vdot_ij[2][0]) + norm(vdot_ij[1][2]) + norm(vdot_ij[2][1]) - norm(vdot_ij[0][0] + vdot_ij[1][1] + vdot_ij[2][2]) / 2.;
            }
        }
    }

    std::vector<double> energy_values = energy_density();
    double total = energy_values.back();

    double rho_GW = 0.;

    hdot2[0] = 0.;
    for (i_GW = 0; i_GW < 3; i_GW++) {
        for (j_GW = 0; j_GW < 3; j_GW++) {
            hdot2[0] += pw2(pi_u_k[i_GW][j_GW][0][0]) + pw2(pi_u_k[i_GW][j_GW][0][1]);
        }
    }
    rho_GW += hdot2[0];

    for (i = 1; i < maxnumbins; i++) {
        p[i] = kIR * i;
        // p[i] = kOverA * i;

        rho_GW += hdot2[i];

        if (numpoints[i] > 0) {
            h2[i]    = h2[i]    / numpoints[i];
            hdot2[i] = hdot2[i] / numpoints[i];
        }


        spectra_GWs_ << std::fixed << std::setprecision(kIR_prec) << p[i] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << i / pw2(N3) * numpoints[i] * h2[i] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << i / pw2(N3) / 4. / pow(a, 6) * pw2(fStar / MPl) / total * numpoints[i] * hdot2[i] << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pw2(MPl * omegaStar) / pow(fStar, 4) * 2 / pw2(pi) * pw2(ad) / pow(a, 2 * alpha + 2) << " " << numpoints[i] << std::endl;
    }

    spectra_GWs_ << std::endl;

    rho_GW *= pw2(omegaStar) * pow(fStar, 4) / pw2(MPl) / 4. / pw2(N3) / pow(a, 6);
    rho_GW /= pw2(fStar * omegaStar);

    GWs_energy_ << std::fixed << std::setprecision(Infreq_prec) << t << " " << std::scientific << std::setprecision(6) << rho_GW / total << " " << rho_GW << std::endl;

    #endif

}




void energy() {
    static std::ofstream energy_, conservation_;
    static double totalinitial; // Initial value of energy density (used for checking conservation in Minkowski space)
    DECLARE_INDICES
    int fld;
    double var, vard, ffd; // Averaged field values over the grid (defined below)
    double deriv_energy, grad_energy, pot_energy, total; // Total gives the total value of rho(t) for checking energy conservation
    double LHS, RHS;

    static bool isFirst = true;
    if (isFirst) { // Open output files (isFirst is set to zero at the bottom of the function)
        name_ = dir_ + "/average_energies" + ext_;
        // if (continue_run)
            energy_.open(name_, std::ios::out | std::ios::app);
        // else
        //     energy_.open(name_, std::ios::out | std::ios::trunc);

        // if (energy_ == nullptr) {
        //     std::cerr << "Failed to open file: " << name_ << std::endl;
        // } else {
        //     std::cout << "File opened successfully: " << name_ << std::endl;
        //     // 在适当的时候关闭文件，例如在程序结束时
        //     fclose(energy_);
        // }
        if (expansion != 1) { // For power-law expansion don't calculate conservation
            name_ = dir_ + "/average_energy_conservation" + ext_;
            // if (continue_run)
                conservation_.open(name_, std::ios::out | std::ios::app);
            // else
            //     conservation_.open(name_, std::ios::out | std::ios::trunc);
        }
    } // The variable isFirst is used again at the end of the function, where it is then set to 0


    #if SPECTRAL_FLAG
    if (isChanged) {
        fftw_plan forward_plan_f;
        fftw_plan forward_plan_pi;
        for (fld = 0; fld < nflds; fld++) {

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
            }

        }
        fftw_destroy_plan(forward_plan_f);
        fftw_destroy_plan(forward_plan_pi);
        isChanged = false;
    }
    #endif


    // total = 0.;
    // // Calculate and output kinetic (time derivative) energy
    // for (fld = 0; fld < nflds; fld++) { // Note that energy is output for all fields regardless of the value of nflds
    //     // deriv_energy = kinetic_energy(a, pi_f[fld]);
    //     #if !SPECTRAL_FLAG
    //         deriv_energy = kinetic_energy(fld);
    //     #else
    //         deriv_energy = kinetic_energy_k(fld);
    //     #endif
    //     total += deriv_energy;
    //     energy_ << std::scientific << std::setprecision(6) << " " << deriv_energy;
    
    //     // grad_energy = gradient_energy(a, f[fld], fld);
    //     #if !SPECTRAL_FLAG
    //         grad_energy = gradient_energy(fld);
    //     #else
    //         grad_energy = gradient_energy_k(fld);
    //     #endif
    //     total += grad_energy;
    //     energy_ << std::scientific << std::setprecision(6) << " " << grad_energy;
    // }

    // // Calculate and output potential energy
    // for (i = 0; i < num_potential_terms; i++) {
    //     pot_energy = potential_energy(i, nullptr); // Model dependent function for calculating potential energy terms
    //     total += pot_energy;
    //     energy_ << std::scientific << std::setprecision(6) << " " << pot_energy;
    // }
    // energy_ << std::scientific << std::setprecision(6) << " " << total << std::endl;


    std::vector<double> energy_values = energy_density();
    total = energy_values.back();

    energy_ << std::fixed << std::setprecision(Freq_prec) << t;
    for (const double& energy : energy_values) {
        energy_ << std::scientific << std::setprecision(6) << " " << energy;
    }
    energy_ << std::endl;


    // Energy conservation
    if (isFirst) { // In Minkowski space record the initial value of the energy to be used for checking energy conservation
        if (expansion == 0)
            totalinitial = total;
        isFirst = false; // Regardless of the expansion set isFirst to 0 so file streams aren't opened again.
    }

    if (expansion != 1) { // Conservation isn't checked for power law expansion
        if (expansion == 0) {// In Minkowski space the file conservation_ records the ratio of rho(t) to rho_initial
            conservation_ << std::fixed << std::setprecision(Freq_prec) << t << " " << std::scientific << std::setprecision(6) << total / totalinitial << std::endl;
        } else { // In an expanding universe the file conservation_ records the ratio of H^2(t) to 8 pi/3 rho(t)
            LHS = pw2(ad);
            RHS = pw2(fStar / MPl) * pow(a, 2 * (alpha + 1)) * total / 3.;
            conservation_ << std::fixed << std::setprecision(Freq_prec) << t << " " << std::scientific << std::setprecision(8) << fabs(LHS - RHS) / fabs(LHS + RHS) << " " << std::fixed << std::setprecision(6) << LHS << " " << RHS << std::endl;
        }
    }
}





// Outputs histograms for all output fields
// The histograms are stored in files "histogram<filenumber>" where <filenumber> runs from 0 to nflds.
// A separate file, "histogramtimes", records the times at which histograms are recorded, the minimum field value for each field, and the spacing (in field values) between successive bins for each time.
void histograms() {
    static std::ofstream histogram_[nflds], histogramtimes_;
    int i = 0, j = 0, k = 0, fld;
    int binnum; // Index of bin for a given field value
    double binfreq[nbins]; // The frequency of field values occurring within each bin
    double bmin, bmax, df; // Minimum and maximum field values for each field and spacing (in field values) between bins
    int numpts; // Count the number of points in the histogram for each field. (Should be all lattice points unless explicit field limits are given.)

    static bool isFirst = true;
    if (isFirst) { // Open output files
        for (fld = 0; fld < nflds; fld++) {
            name_ = dir_ + "/histogram" + std::to_string(fld) + ext_;
            if (continue_run)
                histogram_[fld].open(name_, std::ios::out | std::ios::app);
            else
                histogram_[fld].open(name_, std::ios::out | std::ios::trunc);
        }
        name_ = dir_ + "/histogramtimes" + ext_;
        if (continue_run)
            histogramtimes_.open(name_, std::ios::out | std::ios::app);
        else
            histogramtimes_.open(name_, std::ios::out | std::ios::trunc);
        isFirst = false;
    }


    #if SPECTRAL_FLAG
    if (isChanged) {
        fftw_plan forward_plan_f;
        fftw_plan forward_plan_pi;
        for (fld = 0; fld < nflds; fld++) {

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
            }

        }
        fftw_destroy_plan(forward_plan_f);
        fftw_destroy_plan(forward_plan_pi);
        isChanged = false;
    }
    #endif



    histogramtimes_ << std::fixed << std::setprecision(6) << t;
    
    for (fld = 0; fld < nflds; fld++) { // Main loop. Generate histogram for each field.
        // Find the minimum and maximum values of the field
        if (histogram_max == histogram_min) { // If no explicit limits are given use the current field values
            i = 0; j = 0; k = 0;
            bmin = FIELD(fld);
            bmax = bmin;
            LOOP {
                bmin = (FIELD(fld) < bmin ? FIELD(fld) : bmin);
                bmax = (FIELD(fld) > bmax ? FIELD(fld) : bmax);
            }
        }
        else {
            bmin = histogram_min;
            bmax = histogram_max;
        }

        // Find the difference (in field value) between successive bins
        df = (bmax - bmin) / (double)(nbins); // bmin will be at the bottom of the isFirst bin and bmax at the top of the last

        // Initialize all frequencies to zero
        for (i = 0; i < nbins; i++)
            binfreq[i] = 0.;

        // Iterate over grid to determine bin frequencies
        numpts = 0;
        LOOP {
            binnum = (int)((FIELD(fld) - bmin) / df); // Find index of bin for each value
            if (FIELD(fld) == bmax) // The maximal field value is at the top of the highest bin
                binnum = nbins - 1;
            if (binnum >= 0 && binnum < nbins) { // Increment frequency in the appropriate bin
                binfreq[binnum]++;
                numpts++;
            }
        } // End of loop over grid

        // Output results
        for (i = 0; i < nbins; i++) {
            histogram_[fld] << std::scientific << std::setprecision(6) << binfreq[i] / (double)numpts << std::endl; // Output bin frequency normalized so the total equals 1
        }
            
        histogram_[fld] << std::endl;
        histogramtimes_ << std::scientific << std::setprecision(6) << " " << bmin << " " << df;
    } // End loop over fields

    histogramtimes_ << std::endl;
}

// Calculate two dimensional histograms of pairs of fields
// The histograms are stored in files "histogram2d<filenumber>_<filenumber>".
// A separate file, "histogramtimes", records the times at which histograms are recorded, the minimum field value for each field, and the spacing (in field values) between successive bins for each time.
void histograms_2d() {
    static std::ofstream histogramtimes_;
    static std::vector<std::ofstream> histogram_;
    int i, j, k, pair, fld; // pair iterates over pairs of fields for which histograms are being recorded
    int fld1, fld2; // Indices of the fields for a given histogram
    int binnum[2]; // Indices of bins for a given pair of field values
    double binfreq[nbins0][nbins1]; // The frequency of field values occurring within each bin
    double bmin[2], bmax[2], df[2]; // Minimum and maximum field values for each field and spacing (in field values) between bins

    static bool isFirst = true, numhists; // numhists is the numbers of 2d histograms being calculated
    if (isFirst) { // Open output files
        numhists = (sizeof(hist2dflds) / sizeof(hist2dflds[0]) / 2); // Number of pairs in field list
        histogram_.resize(numhists); // Resize the vector to hold the correct number of ofstream objects
        for (pair = 0; pair < numhists; pair++) {
            name_ = dir_ + "/histogram2d" + std::to_string(hist2dflds[2 * pair]) + "_" + std::to_string(hist2dflds[2 * pair + 1]) + ext_;
            if (continue_run)
                histogram_[pair].open(name_, std::ios::out | std::ios::app);
            else
                histogram_[pair].open(name_, std::ios::out | std::ios::trunc);
        }
        name_ = dir_ + "/histogram2dtimes" + ext_;
        if (continue_run)
            histogramtimes_.open(name_, std::ios::out | std::ios::app);
        else
            histogramtimes_.open(name_, std::ios::out | std::ios::trunc);
        isFirst = false;
    }


    #if SPECTRAL_FLAG
    if (isChanged) {
        fftw_plan forward_plan_f;
        fftw_plan forward_plan_pi;
        for (fld = 0; fld < nflds; fld++) {

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
            }

        }
        fftw_destroy_plan(forward_plan_f);
        fftw_destroy_plan(forward_plan_pi);
        isChanged = false;
    }
    #endif


    // fprintf(histogramtimes_, "%f", t); // Output time at which 2d histograms were recorded
    histogramtimes_ << std::fixed << std::setprecision(6) << t;

    for (pair = 0; pair < numhists; pair++) { // Main loop. Generate histogram for each field pair
        fld1 = hist2dflds[2 * pair]; // Find indices of fields for 2d histogram
        fld2 = hist2dflds[2 * pair + 1];
        // Find the minimum and maximum values of the field
        if (histogram2d_min == histogram2d_max) { // If no explicit limits are given use the current field values
            i = 0; j = 0; k = 0;
            bmin[0] = FIELD(fld1);
            bmax[0] = bmin[0];
            bmin[1] = FIELD(fld2);
            bmax[1] = bmin[1];
            LOOP {
                bmin[0] = (FIELD(fld1) < bmin[0] ? FIELD(fld1) : bmin[0]);
                bmax[0] = (FIELD(fld1) > bmax[0] ? FIELD(fld1) : bmax[0]);
                bmin[1] = (FIELD(fld2) < bmin[1] ? FIELD(fld2) : bmin[1]);
                bmax[1] = (FIELD(fld2) > bmax[1] ? FIELD(fld2) : bmax[1]);
            }
        }
        else {
            bmin[0] = histogram2d_min;
            bmax[0] = histogram2d_max;
            bmin[1] = histogram2d_min;
            bmax[1] = histogram2d_max;
        }

        // Find the difference (in field value) between successive bins
        df[0] = (bmax[0] - bmin[0]) / (double)(nbins0);
        df[1] = (bmax[1] - bmin[1]) / (double)(nbins1);
        
        // Initialize all frequencies to zero
        for (i = 0; i < nbins0; i++)
            for (j = 0; j < nbins1; j++)
                binfreq[i][j] = 0.;
      
        // Iterate over grid to determine bin frequencies
        LOOP {
            binnum[0] = (int)((FIELD(fld1) - bmin[0]) / df[0]); // Find index of bin for each value
            binnum[1] = (int)((FIELD(fld2) - bmin[1]) / df[1]);
            if (FIELD(fld1) == bmax[0])
                binnum[0] = nbins0 - 1; // The maximal field value is at the top of the highest bin
            if (FIELD(fld2) == bmax[1])
                binnum[1] = nbins1 - 1; // The maximal field value is at the top of the highest bin
            if (binnum[0] >= 0 && binnum[0] < nbins0 && binnum[1] >= 0 && binnum[1] < nbins1)
                binfreq[binnum[0]][binnum[1]]++;
        } // End of loop over grid

        // Output results
        for (i = 0; i < nbins0; i++)
            for (j = 0; j < nbins1; j++) {
                histogram_[pair] << std::scientific << std::setprecision(6) << binfreq[i][j] / (double)N3 << std::endl;
            }
                
        histogram_[pair] << std::endl; // Stick a blank line between times to make the file more readable
        histogramtimes_ << " " << std::scientific << std::setprecision(6) << bmin[0] << " " << df[0] << " " << bmin[1] << " " << df[1]; // Output the starting point and stepsize for the bins at each time
    } // End of loop over pairs

    histogramtimes_ << std::endl;
}

// This function outputs the values of the fields    on a slice of the lattice
inline void slices() {
    static std::ofstream slices_[nflds], slicetimes_;
    static int adjusted_slicedim = slicedim; // Number of dimensions to include on slice
    static int final = slicelength * sliceskip;
    int i, j, k, fld;
    int x, y, z;
    int numpts; // Used for keeping track of how many points are being averaged in each output
    double value; // Field value to be output

    static bool isFirst = true;
    if (isFirst) { // Open output files
        for (fld = 0; fld < nflds; fld++) {
            name_ = dir_ + "/slices" + std::to_string(fld) + ext_;
            if (continue_run)
                slices_[fld].open(name_, std::ios::out | std::ios::app);
            else
                slices_[fld].open(name_, std::ios::out | std::ios::trunc);
        }
        name_ = dir_ + "/slicetimes" + ext_;
        if (continue_run)
            slicetimes_.open(name_, std::ios::out | std::ios::app);
        else
            slicetimes_.open(name_, std::ios::out | std::ios::trunc);

        if (adjusted_slicedim > NDIMS)
            adjusted_slicedim = NDIMS;
        if (final > N)
            final = N;
        isFirst = false;
    }


    #if SPECTRAL_FLAG
    if (isChanged) {
        fftw_plan forward_plan_f;
        fftw_plan forward_plan_pi;
        for (fld = 0; fld < nflds; fld++) {

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
            }

        }
        fftw_destroy_plan(forward_plan_f);
        fftw_destroy_plan(forward_plan_pi);
        isChanged = false;
    }
    #endif



    for (fld = 0; fld < nflds; fld++) {
        if (adjusted_slicedim == 1) {
            for (k = 0; k < final; k += sliceskip) {
                if (sliceaverage == 1) { // Average over all "skipped" points
                    value  = 0.;
                    numpts = 0;
                    for (z = k; z < k + sliceskip && z < N; z++) {  
                        value += FIELDPOINT(fld, 0, 0, z);
                        numpts++;
                    }
                    value /= (double)numpts;
                }
                else // ...or just output field values at the sampled points
                    value = FIELDPOINT(fld, 0, 0, k);
                slices_[fld] << std::scientific << std::setprecision(6) << value << std::endl;
            }
        }
        else if (adjusted_slicedim == 2) {
            for (j = 0; j < final; j += sliceskip)
                for (k = 0; k < final; k += sliceskip) {
                    if (sliceaverage == 1) { // Average over all "skipped" points
                        value  = 0.;
                        numpts = 0;
                        for (y = j; y < j + sliceskip && y < N; y++)
                            for (z = k; z < k + sliceskip && z < N; z++) {
                                value += FIELDPOINT(fld, 0, y, z);
                                numpts++;
                            }
                        value /= (double)numpts;
                    }
                    else // ...or just output field values at the sampled points
                        value = FIELDPOINT(fld, 0, j, k);
                    slices_[fld] << std::scientific << std::setprecision(6) << value << std::endl;
                }
        }
        else if (adjusted_slicedim == 3) {
          for (i = 0; i < final; i += sliceskip)
              for (j = 0; j < final; j += sliceskip)
                  for (k = 0; k < final; k += sliceskip) {
                      if (sliceaverage == 1) { // Average over all "skipped" points
                          value  = 0.;
                          numpts = 0;
                          for (x = i; x < i + sliceskip && x < N; x++)
                              for (y = j; y < j + sliceskip && y < N; y++)
                                  for (z = k; z < k + sliceskip && z < N; z++) {
                                      value += FIELDPOINT(fld, x, y, z);
                                      numpts++;
                                  }
                          value *= 1 / (double)numpts;
                      }
                      else // Average over all "skipped" points
                          value = FIELDPOINT(fld, i, j, k);
                      slices_[fld] << std::scientific << std::setprecision(6) << value << std::endl;
                  }
        }
        slices_[fld] << std::endl;
    }
    slicetimes_ << std::fixed << std::setprecision(6) << t << std::endl;
}


inline void snapshots(size_t step) {
    H5std_string name_;
    DECLARE_INDICES
    name_ = dir_ + snap_dir_ + "/data_" + std::to_string(step) + "_snapshot.h5";

    static H5std_string dataset_names[nflds];
    #if SNAPSHOTS_ENERGY
        static H5std_string dataset_names_energy[4];
    #endif
    static hsize_t dimsf[NDIMS];
    static H5::DataSpace dataspace;
    size_t fld;


    #if SPECTRAL_FLAG
    if (isChanged) {
        fftw_plan forward_plan_f;
        fftw_plan forward_plan_pi;
        for (fld = 0; fld < nflds; fld++) {

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
            }

        }
        fftw_destroy_plan(forward_plan_f);
        fftw_destroy_plan(forward_plan_pi);
        isChanged = false;
    }
    #endif


    static bool isFirst = true;
    if (isFirst) {
        for (fld = 0; fld < nflds; ++fld)
            dataset_names[fld] = "/ScalarField_" + std::to_string(fld);

        // dataset_names[nflds] = "/PotentialEnergy";

        #if SNAPSHOTS_ENERGY
            dataset_names_energy[0] = "/PotentialEnergy";
            dataset_names_energy[1] = "/KineticEnergy";
            dataset_names_energy[2] = "/GradientEnergy";
            dataset_names_energy[3] = "/TotalEnergy";

        #endif

        for (fld = 0; fld < NDIMS; fld++)
            dimsf[fld] = N;
        
        dataspace = H5::DataSpace(NDIMS, dimsf);
        isFirst = false;
    }

    H5::H5File file(name_, H5F_ACC_TRUNC);
    H5::DataSet dataset;


    for (fld = 0; fld < nflds; ++fld) {
        dataset = file.createDataSet(dataset_names[fld], H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(f[fld], H5::PredType::NATIVE_DOUBLE);
        dataset.close();
    }

    #if SNAPSHOTS_ENERGY
        kinetic_energy_lattice(a, pi_f);
        gradient_energy_lattice(a, f);
        double points[nflds];
        LOOP {
            for (fld = 0; fld < nflds; ++fld) {
                points[fld] = f[fld][i][j][k];
            }
            pot_energy[i][j][k] = 0.;
            for (int term = 0; term < num_potential_terms; term++) {
                pot_energy[i][j][k] += potential_energy(term, points);
            }
            tot_energy[i][j][k] = pot_energy[i][j][k] + kine_energy[i][j][k] + grad_energy[i][j][k];
        }

        dataset = file.createDataSet(dataset_names_energy[0], H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(pot_energy, H5::PredType::NATIVE_DOUBLE);
        dataset.close();

        dataset = file.createDataSet(dataset_names_energy[1], H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(kine_energy, H5::PredType::NATIVE_DOUBLE);
        dataset.close();

        dataset = file.createDataSet(dataset_names_energy[2], H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(grad_energy, H5::PredType::NATIVE_DOUBLE);
        dataset.close();

        dataset = file.createDataSet(dataset_names_energy[3], H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(tot_energy, H5::PredType::NATIVE_DOUBLE);
        dataset.close();
    #endif


    file.close();
}


// Output an image of all fields and derivatives on the lattice (and a few other variables) to a binary file
void checkpoint() {
    static std::ofstream grid_;
    // The following variables are all used for keeping track of when to close one field value file and begin writing to another.
    static int numtimes = sizeof(store_lattice_times) / sizeof(double); // Number of times at which to switch grid image files
    static int current = 0; // Index of time value at which to switch files
    static int open; // Indicates whether grid image file is open. (It should be unless the program failed to open the file.)
    int itime; // Integer value of the time up to which a file will be used. Used for generating file names.
    std::string filename;

    static bool isFirst = true;
    if (isFirst) {
        if (numtimes > 0 && store_lattice_times[0] == 0.)
            numtimes = 0;
        if (numtimes == 0) { // If no intermediate times are being output simply call the file grid.img
            filename = dir_ + "/grid.img";
            grid_.open(filename, std::ios::out | std::ios::binary);
        }
        else { // Otherwise label file by the final time it will record data at.
            filename = dir_ + "/grid" + std::to_string(static_cast<int>(store_lattice_times[0])) + ".img";
            grid_.open(filename, std::ios::out | std::ios::binary);
        }
        isFirst = false;
    }
    else if (current < numtimes && t > store_lattice_times[current]) { // If one of the times listed above has been passed switch to another field value file.
        if (open) {
            grid_.close();
        }
            
        current++;
        itime = (current < numtimes ? (int)store_lattice_times[current] : (int)tf); // After last time indicated name file with final time of run
        filename = dir_ + "/grid" + std::to_string(itime) + ".img";
        grid_.open(filename, std::ios::out | std::ios::binary);
    }

    if (!grid_.is_open()) {
        std::cout << "Error: Grid checkpointing file not open" << std::endl;
        open = 0;
    }
    else
        open = 1;

    if (open) { // Write a binary image of all current values to the file grid_.
        grid_.seekp(0, std::ios::beg);

        // Write the data to the file
        grid_.write(reinterpret_cast<const char*>(&run_number), sizeof(run_number));
        grid_.write(reinterpret_cast<const char*>(&t), sizeof(t));
        grid_.write(reinterpret_cast<const char*>(&a), sizeof(a));
        grid_.write(reinterpret_cast<const char*>(&ad), sizeof(ad));
        grid_.write(reinterpret_cast<const char*>(f), sizeof(double) * nflds * N3);
        grid_.write(reinterpret_cast<const char*>(pi_f), sizeof(double) * nflds * N3);
        // Ensure all data is written to the file
    }
}


void readable_time(int t, std::ofstream &info_) {
    const int tminutes = 60;
    const int thours = 60 * tminutes;
    const int tdays = 24 * thours;

    if (t == 0) {
        info_ << "less than 1 second\n";
        return;
    }

    bool isFirst = true;

    // Days
    if (t > tdays) {
        info_ << (isFirst ? "" : ", ") << t / tdays << " days";
        t %= tdays;
        isFirst = false;
    }
    // Hours
    if (t > thours) {
        info_ << (isFirst ? "" : ", ") << t / thours << " hours";
        t %= thours;
        isFirst = false;
    }
    // Minutes
    if (t > tminutes) {
        info_ << (isFirst ? "" : ", ") << t / tminutes << " minutes";
        t %= tminutes;
        isFirst = false;
    }
    // Seconds
    if (t > 0) {
        info_ << (isFirst ? "" : ", ") << t << " seconds";
    }
    info_ << "\n";
    return;
}

/////////////////////////////////////////////////////
// Externally called function(s)
/////////////////////////////////////////////////////

void output_parameters() {
    static std::ofstream info_;
    static std::time_t tStart, tFinish; // Keep track of elapsed clock time

    static bool isFirst = true;
    if (isFirst) { // At beginning of run output run parameters
        std::string name_ = std::string(dir_) + "/info" + ext_;
        info_.open(name_, std::ios::out);

        if (!info_.is_open()) {
            std::cerr << "Unable to open file: " << name_ << std::endl;
            return;
        }

        info_ << "--------------------------\n";
        info_ << "Model Specific Information\n";
        info_ << "--------------------------\n";
        modelinfo_(info_);

        info_ << "\n--------------------------\n";
        info_ << "General Program Information\n";
        info_ << "-----------------------------\n";
        info_ << "Grid size=" << N << "^" << NDIMS << "\n";
        info_ << "Number of fields=" << nflds << "\n";
        info_ << "L=" << L << "\n";
        info_ << "kIR=" << kIR << "\n";
        info_ << std::setprecision(6) << "dt=" << dt << ", dt/dx=" << dt / dx << "\n";
        if (expansion == 0)
            info_ << "No expansion\n";
        else if (expansion == 1)
            info_ << "Fixed background expansion\n";
        else if (expansion == 2)
            info_ << "Expansion calculated self-consistently\n";

        std::time(&tStart);
        info_ << "\nRun began at " << std::ctime(&tStart); // Output date in readable form
        isFirst = false;
    } else { // If not at beginning record elapsed time for run
        std::time(&tFinish);
        info_ << "Run ended at " << std::ctime(&tFinish); // Output ending date
        info_ << "\nRun from t=" << t0 << " to t=" << t << " took ";
        readable_time(static_cast<int>(tFinish - tStart), info_);
        info_ << std::endl;
    }
}

void save() {
    // Model-specific output
    if (smodel) // Allows each model to define its own output function(s) and set rescaling for other outputs
        model_output(ext_);

    // if (smeansvars) // Calculate means and variances of fields
    meansvars();

    if (expansion == 2) // Output scale factor and its derivatives
        scale();

    // Infrequent calculations
    // if (senergy && t >= tenergy) // Calculate all contributions to energy density
    energy();

    if (shistograms) // Calculate histograms of all output fields
        histograms();

    if (shistograms2d) // Calculate two dimensional histograms of pairs of fields
        histograms_2d();

    if (sslices) // Calculate two dimensional histograms of pairs of fields
        slices();

    if (ssnapshot) // Calculate two dimensional histograms of pairs of fields
        snapshots(numsteps);

    // if (sspectra && t >= tspectra) // Calculate and power spectra of all output fields
    spectra();

    if (scheckpoint) // Save an image of the grid.
        checkpoint();

}

void save_Freq() {
    // if (smeansvars) // Calculate means and variances of fields
    meansvars();

    if (expansion == 2) // Output scale factor and its derivatives
        scale();

    // if (senergy && t >= tenergy) // Calculate all contributions to energy density
    energy();
}

void save_Infreq() {
    if (shistograms) // Calculate histograms of all output fields
        histograms();

    // if (sspectra && t >= tspectra) // Calculate and power spectra of all output fields
    spectra();
}

void save_Snap(double numsteps) {
    if (ssnapshot)
        snapshots(numsteps);
}

#if ANALYTIC_EVOLUTION
void save_analytic() {
    // Model-specific output
    if (smodel) // Allows each model to define its own output function(s) and set rescaling for other outputs
        model_output(ext_);

    // if (smeansvars) // Calculate means and variances of fields
    meansvars_analytic();

    if (sexpansion && expansion == 2) // Output scale factor and its derivatives
        scale_analytic();

    // Infrequent calculations
    // if (senergy && t >= tenergy) // Calculate all contributions to energy density
    energy_analytic();

    // if (sspectra && t >= tspectra) // Calculate and power spectra of all output fields
    spectra_analytic_discrete();
}

void save_Freq_analytic() {
    // if (smeansvars) // Calculate means and variances of fields
    meansvars_analytic();

    if (expansion == 2) // Output scale factor and its derivatives
        scale_analytic();

    // if (senergy && t >= tenergy) // Calculate all contributions to energy density
    energy_analytic();
}

void save_Infreq_analytic() {
    // if (shistograms && t >= thistograms) // Calculate histograms of all output fields
    //     histograms();

    // if (sspectra && t >= tspectra) // Calculate and power spectra of all output fields
    spectra_analytic_discrete();
}

#endif





#if BIFURCATION


void meansvars_bifurcation() {
    static std::ofstream trajectory_;
    DECLARE_INDICES
    size_t fld;

    static bool isFirst = true;
    if (isFirst) {
        name_ = dir_ + "/average_scalar_bifurcation_" + ext_;
        // if (continue_run)
            trajectory_.open(name_, std::ios::out | std::ios::app);
        // else
        //     trajectory_.open(name_, std::ios::out | std::ios::trunc);
        isFirst = false;
    }

    #if SPECTRAL_FLAG
    if (isChanged) {
        for (fld = 0; fld < nflds; fld++) {
            fftw_plan forward_plan_f;
            fftw_plan forward_plan_pi;

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
            }

            fftw_destroy_plan(forward_plan_f);
            fftw_destroy_plan(forward_plan_pi);
        }
        isChanged = false;
    }
    #endif

    for (fld = 0; fld < nflds; fld++) {
        f_av_positive[fld] = 0.;
        f_av_negative[fld] = 0.;
    }

    LOOP {
        for (fld = 0; fld < nflds; fld++) {
            f_av_positive[fld] += f_positive[fld][i][j][k];
            f_av_negative[fld] += f_negative[fld][i][j][k];
        }
    }
    
    for (fld = 0; fld < nflds; fld++) {
        f_av_positive[fld] /= count_positive;
        f_av_negative[fld] /= N3 - count_positive;
    }

    trajectory_  << std::fixed << std::setprecision(Freq_prec) << t << std::scientific << std::setprecision(6);
    for (fld = 0; fld < nflds; fld++) {
        trajectory_ << std::setw(setw_leng) << f_av_positive[fld] << std::setw(setw_leng) << f_av_negative[fld];
    }
    trajectory_  << std::endl;
}


#if ANALYTIC_EVOLUTION
void spectra_trajectory_analytic() {
    DECLARE_INDICES
    static std::ofstream spectra_positive_[nflds], spectra_negative_[nflds];

    static std::ofstream spectra_positive_Evolution_[nflds], spectra_negative_Evolution_[nflds];

    static std::ofstream R_[nflds];

    size_t fld;


    double k_mode;
    double fk2[nflds];
    // double pik2[nflds];
    // double positive_fk2[nflds], negative_fk2[nflds];

    static std::vector<std::vector<double>> R_i(nflds, std::vector<double>(maxnumbins));

    static bool isFirst = true;
    if (isFirst) {
        for (fld = 0; fld < nflds; fld++) {
            name_ = dir_ + "/analytic_spectra_positve_scalar_" + std::to_string(fld) + ext_;
            if (continue_run)
                spectra_positive_[fld].open(name_, std::ios::out | std::ios::app);
            else
                spectra_positive_[fld].open(name_, std::ios::out | std::ios::trunc);


            name_ = dir_ + "/analytic_spectra_negative_scalar_" + std::to_string(fld) + ext_;
            if (continue_run)
                spectra_negative_[fld].open(name_, std::ios::out | std::ios::app);
            else
                spectra_negative_[fld].open(name_, std::ios::out | std::ios::trunc);


            name_ = dir_ + "/analytic_spectra_positive_scalar_" + std::to_string(fld) + "_evolution" + ext_;
            if (continue_run)
                spectra_positive_Evolution_[fld].open(name_, std::ios::out | std::ios::app);
            else
                spectra_positive_Evolution_[fld].open(name_, std::ios::out | std::ios::trunc);


            name_ = dir_ + "/analytic_spectra_negative_scalar_" + std::to_string(fld) + "_evolution" + ext_;
            if (continue_run)
                spectra_negative_Evolution_[fld].open(name_, std::ios::out | std::ios::app);
            else
                spectra_negative_Evolution_[fld].open(name_, std::ios::out | std::ios::trunc);


            name_ = dir_ + "/R_scalar_" + std::to_string(fld) + "_evolution" + ext_;
            if (continue_run)
                R_[fld].open(name_, std::ios::out | std::ios::app);
            else
                R_[fld].open(name_, std::ios::out | std::ios::trunc);

            for (i = 0; i < maxnumbins; i++) {
                spectra_negative_Evolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " ";
                spectra_positive_Evolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " ";
                R_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " ";


                auto it = k_square.find(pw2(i));
                size_t position = std::distance(k_square.begin(), it);

                R_i[fld][i] = sqrt(pw2(fk_left[fld][position].real()) + pw2(fk_left[fld][position].imag())) * pow(a_k, 3 - alpha) * sqrt(pw2(ad_k / a_k * fk_left[fld][position].real() + pow(a_k, alpha - 3) * pi_k_left[fld][position].real()) + pw2(ad_k / a_k * fk_left[fld][position].imag() + pow(a_k, alpha - 3) * pi_k_left[fld][position].imag()));
            }

            spectra_positive_Evolution_[fld] << std::endl;
            spectra_negative_Evolution_[fld] << std::endl;
            R_[fld] << std::endl;
        }

        isFirst = false;
    }


    for (fld = 0; fld < nflds; fld++) {

        spectra_positive_Evolution_[fld] << std::fixed << std::setprecision(6) << t << " ";
        spectra_negative_Evolution_[fld] << std::fixed << std::setprecision(6) << t << " ";

        R_[fld] << std::fixed << std::setprecision(6) << t << " ";


        for (k = 1; k < maxnumbins; k++) {
            auto it = k_square.find(pw2(k));
            size_t position = std::distance(k_square.begin(), it);
            k_mode = sqrt(k_square_vec[position]) * kIR;

            fk2[fld] = pw2(fk_left[fld][position].real()) + pw2(fk_left[fld][position].imag());

            spectra_positive_[fld] << std::fixed << std::setprecision(kIR_prec) << k_mode << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(k_mode, 3) / (2 * pw2(M_PI)) * fk2[fld] << std::endl;
            spectra_negative_[fld] << std::fixed << std::setprecision(kIR_prec) << k_mode << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(k_mode, 3) / (2 * pw2(M_PI)) * fk2[fld] << std::endl;



            spectra_positive_Evolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << sqrt(fk2[fld]) << " ";
            spectra_negative_Evolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << sqrt(fk2[fld]) << " ";

            R_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << R_i[fld][k] / (sqrt(pw2(fk_left[fld][position].real()) + pw2(fk_left[fld][position].imag())) * pow(a_k, 3 - alpha) * sqrt(pw2(ad_k / a_k * fk_left[fld][position].real() + pow(a_k, alpha - 3) * pi_k_left[fld][position].real()) + pw2(ad_k / a_k * fk_left[fld][position].imag() + pow(a_k, alpha - 3) * pi_k_left[fld][position].imag()))) << " ";
        }

        spectra_positive_[fld] << std::endl;
        spectra_negative_[fld] << std::endl;


        spectra_positive_Evolution_[fld] << std::endl;
        spectra_negative_Evolution_[fld] << std::endl;

        R_[fld] << std::endl;
    }
}
#endif


void spectra_trajectory_lattice() {
    DECLARE_INDICES
    static std::ofstream spectra_positive_[nflds], spectra_negative_[nflds];

    static std::ofstream spectra_positive_Evolution_[nflds], spectra_negative_Evolution_[nflds];

    size_t fld;


    double k_mode;
    // double phi_postive_spectra, phi_negative_spectra, chi_postive_spectra, chi_negative_spectra;
    double positive_fk2[nflds], negative_fk2[nflds];
    // double positive_pik2[nflds], negative_pik2[nflds];

    static bool isFirst = true;
    if (isFirst) {
        for (fld = 0; fld < nflds; fld++) {
            name_ = dir_ + "/analytic_spectra_positve_scalar_" + std::to_string(fld) + ext_;
            // if (continue_run)
                spectra_positive_[fld].open(name_, std::ios::out | std::ios::app);
            // else
            //     spectra_positive_[fld].open(name_, std::ios::out | std::ios::trunc);

            name_ = dir_ + "/analytic_spectra_negative_scalar_" + std::to_string(fld) + ext_;
            // if (continue_run)
                spectra_negative_[fld].open(name_, std::ios::out | std::ios::app);
            // else
            //     spectra_negative_[fld].open(name_, std::ios::out | std::ios::trunc);

            name_ = dir_ + "/analytic_spectra_positive_scalar_" + std::to_string(fld) + "_evolution" + ext_;
            spectra_positive_Evolution_[fld].open(name_, std::ios::out | std::ios::app);

            name_ = dir_ + "/analytic_spectra_negative_scalar_" + std::to_string(fld) + "_evolution" + ext_;
            spectra_negative_Evolution_[fld].open(name_, std::ios::out | std::ios::app);

            #if !ANALYTIC_EVOLUTION
                // spectra_positive_Evolution_[fld] << 0 << " ";
                // spectra_negative_Evolution_[fld] << 0 << " ";
                for (i = 0; i < maxnumbins; i++) {
                    spectra_positive_Evolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " ;
                    spectra_negative_Evolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " ;

                    // spectra_positive_Evolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " ";
                    // spectra_negative_Evolution_[fld] << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " " << std::fixed << std::setprecision(kIR_prec) << kIR * i << " ";
                }
                spectra_positive_Evolution_[fld] << std::endl;
                spectra_negative_Evolution_[fld] << std::endl;
            #endif
        }

        isFirst = false;
    }


    for (fld = 0; fld < nflds; fld++) {

        spectra_positive_Evolution_[fld] << std::fixed << std::setprecision(6) << t << " ";
        spectra_negative_Evolution_[fld] << std::fixed << std::setprecision(6) << t << " ";


        for (k = 1; k < maxnumbins; k++) {
            k_mode = k * kIR;
            // k_mode = k * kOverA;

            positive_fk2[fld] = pw2(fk_positive[fld][k].real()) + pw2(fk_positive[fld][k].imag());
            negative_fk2[fld] = pw2(fk_negative[fld][k].real()) + pw2(fk_negative[fld][k].imag());

            // positive_pik2[fld] = pw2(pi_k_positive[fld][k].real()) + pw2(pi_k_positive[fld][k].imag());
            // negative_pik2[fld] = pw2(pi_k_negative[fld][k].real()) + pw2(pi_k_negative[fld][k].imag());


            spectra_positive_[fld] << std::fixed << std::setprecision(kIR_prec) << k_mode << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(k_mode, 3) / (2 * pw2(M_PI)) * positive_fk2[fld] << std::endl;
            spectra_negative_[fld] << std::fixed << std::setprecision(kIR_prec) << k_mode << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(k_mode, 3) / (2 * pw2(M_PI)) * negative_fk2[fld] << std::endl;


            spectra_positive_Evolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << sqrt(positive_fk2[fld]) << " ";
            spectra_negative_Evolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << sqrt(negative_fk2[fld]) << " ";
            // spectra_positive_Evolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << sqrt(positive_fk2[fld]) << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << fk_positive[fld][k].real() << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << fk_positive[fld][k].imag() << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(a, alpha - 3) * sqrt(positive_pik2[fld]) << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(a, alpha - 3) * pi_k_positive[fld][k].real() << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(a, alpha - 3) * pi_k_positive[fld][k].imag() << " ";
            // spectra_negative_Evolution_[fld] << std::scientific << std::setprecision(6) << std::setw(setw_leng) << sqrt(negative_fk2[fld]) << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << fk_negative[fld][k].real() << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << fk_negative[fld][k].imag() << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(a, alpha - 3) * sqrt(negative_pik2[fld]) << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(a, alpha - 3) * pi_k_negative[fld][k].real() << " " << std::scientific << std::setprecision(6) << std::setw(setw_leng) << pow(a, alpha - 3) * pi_k_negative[fld][k].imag() << " ";
        }

        spectra_positive_[fld] << std::endl;
        spectra_negative_[fld] << std::endl;


        spectra_positive_Evolution_[fld] << std::endl;
        spectra_negative_Evolution_[fld] << std::endl;
    }
}

#endif



#if BIFURCATION
void d2Vdf2_evolution() {
    static std::ofstream d2Vdf2_;

    static bool isFirst = true;
    if (isFirst) {
        name_ = dir_ + "/d2Vdf2_evolution" + ext_;
        if (continue_run)
            d2Vdf2_.open(name_, std::ios::out | std::ios::app);
        else
            d2Vdf2_.open(name_, std::ios::out | std::ios::trunc);
        
        isFirst = false;
    }

    d2Vdf2_ << std::fixed << std::setprecision(Freq_prec) << t << std::scientific << std::setprecision(6);
    for (size_t term = 0; term < nflds + 1; term++) {
        d2Vdf2_ << std::setw(setw_leng) << d2Vdf2(term, f_av_positive) << std::setw(setw_leng) << d2Vdf2(term, f_av_negative);
    }

    double negative_f_av_positive[nflds];
    negative_f_av_positive[0] = f_av_positive[0];
    negative_f_av_positive[1] = -f_av_positive[1];

    for (size_t term = 0; term < nflds + 1; term++) {
        d2Vdf2_ << std::setw(setw_leng) << d2Vdf2(term, negative_f_av_positive);
    }

    d2Vdf2_ << std::endl;
}

#endif