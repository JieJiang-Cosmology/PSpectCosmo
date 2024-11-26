#include "pspectcosmo.h"
#include <array>
#include <vector>



void kinetic_energy_lattice(double a, const double (&pi_f)[nflds][N][N][N]) {
    DECLARE_INDICES
    double norm = pow(a, 6) / 2.;
    size_t fld;

    LOOP {
        kine_energy[i][j][k] = 0.;
        for (fld = 0; fld < nflds; fld++) {
            kine_energy[i][j][k] += pw2(pi_f[fld][i][j][k]);
        }
        kine_energy[i][j][k] *= norm;
    }
}



void gradient_energy_lattice(double a, double (&f)[nflds][N][N][N]) {
    DECLARE_INDICES
    int howmany;
    int idist, odist;
    int istride, ostride;
    double temp;


    fftw_complex *fft_result = fftw_alloc_complex(N * N * (N/2+1));
    fftw_plan plan;

    LOOP
        grad_energy[i][j][k] = 0.;

    for (int fld = 0; fld < nflds; fld++) {

        howmany = N * N;
        idist = 1, odist = N/2+1;
        istride = N * N, ostride = 1;

        plan = fftw_plan_many_dft_r2c(1, &N, howmany, &f[fld][0][0][0], NULL, istride, idist, fft_result, NULL, ostride, odist, FFTW_ESTIMATE);
        fftw_execute(plan);

        for (i = 0; i < N; ++i) {
            for (j = 0; j < N; ++j) {
                for (k = 0; k < N/2; ++k) {
                    temp = fft_result[(i * N + j) * (N/2+1) + k][0];
                    fft_result[(i * N + j) * (N/2+1) + k][0] = - kIR * k * fft_result[(i * N + j) * (N/2+1) + k][1];
                    fft_result[(i * N + j) * (N/2+1) + k][1] = kIR * k * temp;
                }
                fft_result[(i * N + j) * (N/2+1) + N/2][0] = 0.;
                fft_result[(i * N + j) * (N/2+1) + N/2][1] = 0.;
            }
        }

        plan = fftw_plan_many_dft_c2r(1, &N, howmany, fft_result, NULL, ostride, odist, &data_1st_deri[0][0][0], NULL, istride, idist, FFTW_ESTIMATE);
        fftw_execute(plan);

        for (i = 0; i < N; i++)
            for (j = 0; j < N; ++j)
                for (k = 0; k < N; ++k) {
                    data_1st_deri[i][j][k] /= N;
                    grad_energy[i][j][k] += pw2(data_1st_deri[i][j][k]);
                }

        howmany = N;
        idist = 1, odist = N/2+1;
        istride = N, ostride = 1;

        for (i = 0; i < N; i++) {
            plan = fftw_plan_many_dft_r2c(1, &N, howmany, &f[fld][i][0][0], NULL, istride, idist, fft_result + i * N * (N/2+1), NULL, ostride, odist, FFTW_ESTIMATE);
            fftw_execute(plan);
        }

        for (i = 0; i < N; ++i) {
            for (j = 0; j < N; ++j) {
                for (k = 0; k < N/2; ++k) {
                    temp = fft_result[(i * N + j) * (N/2+1) + k][0];
                    fft_result[(i * N + j) * (N/2+1) + k][0] = - kIR * k * fft_result[(i * N + j) * (N/2+1) + k][1];
                    fft_result[(i * N + j) * (N/2+1) + k][1] = kIR * k * temp;
                }
                fft_result[(i * N + j) * (N/2+1) + N/2][0] = 0.;
                fft_result[(i * N + j) * (N/2+1) + N/2][1] = 0.;
            }
        }

        for (i = 0; i < N; i++) {
            plan = fftw_plan_many_dft_c2r(1, &N, howmany, fft_result + i * N * (N/2+1), NULL, ostride, odist, &data_1st_deri[i][0][0], NULL, istride, idist, FFTW_ESTIMATE);
            fftw_execute(plan);
        }

        for (i = 0; i < N; i++)
            for (j = 0; j < N; ++j)
                for (k = 0; k < N; ++k) {
                    data_1st_deri[i][j][k] /= N;
                    grad_energy[i][j][k] += pw2(data_1st_deri[i][j][k]);
                }

        howmany = N * N;
        idist = N, odist = N/2+1;
        istride = 1, ostride = 1;

        plan = fftw_plan_many_dft_r2c(1, &N, howmany, &f[fld][0][0][0], NULL, istride, idist, fft_result, NULL, ostride, odist, FFTW_ESTIMATE);
        fftw_execute(plan);

        for (i = 0; i < N; ++i) {
            for (j = 0; j < N; ++j) {
                for (k = 0; k < N/2; ++k) {
                    temp = fft_result[(i * N + j) * (N/2+1) + k][0];
                    fft_result[(i * N + j) * (N/2+1) + k][0] = - kIR * k * fft_result[(i * N + j) * (N/2+1) + k][1];
                    fft_result[(i * N + j) * (N/2+1) + k][1] = kIR * k * temp;
                }
                fft_result[(i * N + j) * (N/2+1) + N/2][0] = 0.;
                fft_result[(i * N + j) * (N/2+1) + N/2][1] = 0.;
            }
        }

        plan = fftw_plan_many_dft_c2r(1, &N, howmany, fft_result, NULL, ostride, odist, &data_1st_deri[0][0][0], NULL, istride, idist, FFTW_ESTIMATE);
        fftw_execute(plan);

        for (i = 0; i < N; i++)
            for (j = 0; j < N; ++j)
                for (k = 0; k < N; ++k) {
                    data_1st_deri[i][j][k] /= N;
                    grad_energy[i][j][k] += pw2(data_1st_deri[i][j][k]);
                }
        
    }

    double norm = pw2(a) / 2.;
    LOOP
        grad_energy[i][j][k] *= norm;

    fftw_destroy_plan(plan);
    fftw_free(fft_result);
}






double kernel_a_only_backgroud(const double a, double (&f)[nflds], const double (&pi_f)[nflds]) {
    double kernel_a;
    double K_E = 0;
    double G_E = 0;
    double V_E = 0;

    for (size_t fld = 0; fld < nflds; fld++) {
        K_E += pw2(pi_f[fld]);
    }
    K_E /=  2. * pow(a, 6);

    for (size_t term = 0; term < num_potential_terms; term++) {
        V_E += potential_energy(term, f);
    }

    kernel_a = pow(a, 1 + 2 * alpha) / 3. * pw2(fStar / MPl) * ((alpha - 2) * K_E + alpha * G_E + (alpha + 1) * V_E);
    return kernel_a;
}


#if ANALYTIC_EVOLUTION

    void kernel_field_background(const double a, double (&f)[nflds], double (&kernel_field)[nflds]) {
        double norm2 = pow(a, 3 + alpha);

        for (size_t fld = 0; fld < nflds; fld++)
            kernel_field[fld] =  - norm2 * dvdf(fld, f);;
    }

    void evolve_VV_analytic_perturbation(double dt) {
        DECLARE_INDICES
        double b_k, b_half_k, a_half_k, a_of_pi_k;
        size_t fld;
        double k_mode;
        double k_square;

        std::complex<double> pi_k_left_half[nflds][num_k_values];
        std::complex<double> pi_k_right_half[nflds][num_k_values];

        double kernel_f_left[nflds], kernel_f_right[nflds];
        double f_left_bg[nflds], f_right_bg[nflds];
        double pi_left_bg[nflds], pi_right_bg[nflds];

        double norm1_k, norm2_k, norm3_k;

        b_k = ad_k;
        

        for (size_t p = 0; p < s; p++) {
            // step 1
            b_half_k = b_k + w[p] * dt / 2. * add_k;

            // step 2, analytic background
            for (fld = 0; fld < nflds; fld++) {
                f_left_bg[fld]  = fk_left[fld][0].real();
                f_right_bg[fld] = fk_right[fld][0].real();
            }

            kernel_field_background(a_k, f_left_bg, kernel_f_left);
            kernel_field_background(a_k, f_right_bg, kernel_f_right);


            for (fld = 0; fld < nflds; fld++) {
                // background half value is imaginary part, (analytic background)
                pi_k_left[fld][0].imag(
                    pi_k_left[fld][0].real() + w[p] * dt / 2 * kernel_f_left[fld]
                );
                pi_k_right[fld][0].imag(
                    pi_k_right[fld][0].real() + w[p] * dt / 2 * kernel_f_right[fld]
                );
            }

            // step 1 of perturbation
            norm1_k = pow(a_k, (1 + alpha));
            norm2_k = pow(a_k, (3 + alpha));


            for (k = 1; k < num_k_values; k++) {
                k_square = k_square_vec[k] * pw2(kIR);
                for (fld = 0; fld < nflds; fld++) {
                    pi_k_left_half[fld][k] = 0.;
                    pi_k_right_half[fld][k] = 0.;

                    for (size_t fld_2 = 0; fld_2 < nflds; fld_2++) {
                        pi_k_left_half[fld][k] += d2Vdf2(fld + fld_2, f_left_bg) * fk_left[fld_2][k];
                        pi_k_right_half[fld][k] += d2Vdf2(fld + fld_2, f_right_bg) * fk_right[fld_2][k];
                    }

                    pi_k_left_half[fld][k] *= - norm2_k;
                    pi_k_right_half[fld][k] *= - norm2_k;

                    pi_k_left_half[fld][k] -= norm1_k * k_square * fk_left[fld][k];
                    pi_k_right_half[fld][k] -= norm1_k * k_square * fk_right[fld][k];

                    pi_k_left_half[fld][k] *= w[p] * dt / 2.;
                    pi_k_right_half[fld][k] *= w[p] * dt / 2.;

                    pi_k_left_half[fld][k] += pi_k_left[fld][k];
                    pi_k_right_half[fld][k] += pi_k_right[fld][k];
                }
            }

            // step 3
            a_half_k = a_k + b_half_k * w[p] * dt / 2.;

            // step 4
            norm3_k = pow(a_half_k, alpha - 3);
            for (fld = 0; fld < nflds; fld++) {
                fk_left[fld][0].real(
                    fk_left[fld][0].real() + w[p] * dt * norm3_k * pi_k_left[fld][0].imag()
                );
                fk_right[fld][0].real(
                    fk_right[fld][0].real() + w[p] * dt * norm3_k * pi_k_right[fld][0].imag()
                );

                f_left_bg[fld] = fk_left[fld][0].real();
                f_right_bg[fld] = fk_right[fld][0].real();
            }


            // step 2 of perturbation
            for (k = 1; k < num_k_values; k++) {
                for (fld = 0; fld < nflds; fld++) {
                    fk_left[fld][k] += w[p] * dt * norm3_k * pi_k_left_half[fld][k];
                    fk_right[fld][k] += w[p] * dt * norm3_k * pi_k_right_half[fld][k];
                }
            }

            // step 5
            a_k = a_half_k + b_half_k * w[p] * dt / 2.;

            // step 6, analytic background
            kernel_field_background(a_k, f_left_bg, kernel_f_left);
            kernel_field_background(a_k, f_right_bg, kernel_f_right);

            for (size_t fld = 0; fld < nflds; fld++) {
                // analytic background
                pi_k_left[fld][0].real(
                    pi_k_left[fld][0].imag() + w[p] * dt / 2 * kernel_f_left[fld]
                );
                pi_k_right[fld][0].real(
                    pi_k_right[fld][0].imag() + w[p] * dt / 2 * kernel_f_right[fld]
                );

                pi_left_bg[fld] = pi_k_left[fld][0].real();
                pi_right_bg[fld] = pi_k_right[fld][0].real();
            }


            // step 3 of perturbation
            norm1_k = pow(a_k, (1 + alpha));
            norm2_k = pow(a_k, (3 + alpha));

            for (k = 1; k < num_k_values; k++) {
                k_square = k_square_vec[k] * pw2(kIR);

                for (fld = 0; fld < nflds; fld++) {
                    pi_k_left[fld][k] = 0.;
                    pi_k_right[fld][k] = 0.;

                    for (size_t fld_2 = 0; fld_2 < nflds; fld_2++) {
                        pi_k_left[fld][k] += d2Vdf2(fld + fld_2, f_left_bg) * fk_left[fld_2][k];
                        pi_k_right[fld][k] += d2Vdf2(fld + fld_2, f_right_bg) * fk_right[fld_2][k];
                    }

                    pi_k_left[fld][k] *= - norm2_k;
                    pi_k_right[fld][k] *= - norm2_k;

                    pi_k_left[fld][k] -= norm1_k * k_square * fk_left[fld][k];
                    pi_k_right[fld][k] -= norm1_k * k_square * fk_right[fld][k];


                    pi_k_left[fld][k] *= w[p] * dt / 2.;
                    pi_k_right[fld][k] *= w[p] * dt / 2.;

                    pi_k_left[fld][k] += pi_k_left_half[fld][k];
                    pi_k_right[fld][k] += pi_k_right_half[fld][k];
                }
            }

            // step 7
            add_k = kernel_a_only_backgroud(a_k, f_left_bg, pi_left_bg);
            b_k = b_half_k + w[p] * dt / 2. * add_k;
        }

        ad_k = b_k;

        a = a_k;
        ad = ad_k;
        add = add_k;
        t += dt;
    }

#endif


#if !SPECTRAL_FLAG

    void lapl(double (&f)[N][N][N], double (&f_lapl)[N][N][N]) {
        DECLARE_INDICES

        fftw_complex *fft_result = fftw_alloc_complex(N * N * (N/2+1));
        fftw_plan plan;

        plan = fftw_plan_dft_r2c_3d(N, N, N, &f[0][0][0], fft_result, FFTW_ESTIMATE);
        fftw_execute(plan);

        for (int i = 0; i < N; ++i) {
            int px = (i <= N/2) ? i : i - N;
            for (int j = 0; j < N; ++j) {
                int py = (j <= N/2) ? j : j - N;
                for (int k = 0; k < N/2+1; ++k){
                    int pz = k;
                    fft_result[(i * N + j) * (N/2+1) + k][0] *= (-pw2(kIR * px) - pw2(kIR * py) - pw2(kIR * pz));
                    fft_result[(i * N + j) * (N/2+1) + k][1] *= (-pw2(kIR * px) - pw2(kIR * py) - pw2(kIR * pz));
                }
            }
        }

        fftw_plan backward_plan = fftw_plan_dft_c2r_3d(N, N, N, fft_result, &f_lapl[0][0][0], FFTW_ESTIMATE);
        fftw_execute(backward_plan);

        fftw_destroy_plan(plan);
        fftw_free(fft_result);

        LOOP
            f_lapl[i][j][k] /= N3;
    }


    double kinetic_energy(size_t fld) {
        DECLARE_INDICES
        double kinetic = 0.;

        LOOP
            kinetic += pw2(pi_f[fld][i][j][k]);

        return(.5 * kinetic / pow(a, 6) / (double)N3);
    }

    double gradient_energy(size_t fld) {
        DECLARE_INDICES
        int px, py, pz;

        plan = fftw_plan_dft_r2c_3d(N, N, N, &f[fld][0][0][0], f_k[fld], FFTW_ESTIMATE);
        fftw_execute(plan);

        double grad = 0.;
        for (k = 1; k < N/2; k++) {
            pz = k;
            for (i = 0; i < N; i++) {
                px = (i <= N/2) ? i : i - N;
                for (j = 0; j < N; j++) {
                    py = (j <= N/2) ? j : j - N;
                    grad += pow(kIR, 2) * (px * px + py * py + pz * pz) * (pow(f_k[fld][(i * N + j) * (N/2+1) + k][0], 2) + pow(f_k[fld][(i * N + j) * (N/2+1) + k][1], 2));
                }
            }
        }
        grad *= 2.;
        for (k = 0; k <= N/2; k+=N/2) {
            pz = k;
            for (i = 0; i < N; i++) {
                px = (i <= N/2) ? i : i - N;
                for (j = 0; j < N; j++) {
                    py = (j <= N/2) ? j : j - N;
                    grad += pow(kIR, 2) * (px * px + py * py + pz * pz) * (pow(f_k[fld][(i * N + j) * (N/2+1) + k][0], 2) + pow(f_k[fld][(i * N + j) * (N/2+1) + k][1], 2));
                }
            }
        }
        grad /= N3;
        grad /= 2.;

        return(grad / pw2(a) / (double)N3);
    }


    double kernel_a() {
        double kernel_a;
        double K_E = 0;
        double G_E = 0;
        double V_E = 0;

        for (size_t fld = 0; fld < nflds; fld++) {
            K_E += kinetic_energy(fld);

            G_E += gradient_energy(fld);
        }

        for (int term = 0; term < num_potential_terms; term++) {
            V_E += potential_energy(term, NULL);
        }

        kernel_a = 1. / 3. * pw2(fStar / MPl) * pow(a, 1 + 2 * alpha) * ((alpha - 2) * K_E + alpha * G_E + (alpha + 1) * V_E);
        return kernel_a;
    }

    Array3D kernel_field(size_t term) {
        DECLARE_INDICES

        Array3D kernel_field = {{{0.0}}};

        size_t fld;
        double f_point[nflds];

        double norm1  = pow(a, 1 + alpha);
        double norm2 = pow(a, 3 + alpha);

        lapl(f[term], f_laplacian[term]);

        LOOP {
            for (fld = 0; fld < nflds; fld++)
                f_point[fld] = f[fld][i][j][k];
            kernel_field[i][j][k] = norm1 * f_laplacian[term][i][j][k] - norm2 * dvdf(term, f_point);
        }

        return kernel_field;
    }


#if WITH_GW

    void field_1st_derivative() {
        DECLARE_INDICES
        int howmany;
        int idist, odist;
        int istride, ostride;
        double temp;


        fftw_complex *fft_result = fftw_alloc_complex(N * N * (N/2+1));
        fftw_plan plan;


        for (size_t fld = 0; fld < nflds; fld++) {

            howmany = N * N;
            idist = 1, odist = N/2+1;
            istride = N * N, ostride = 1;

            plan = fftw_plan_many_dft_r2c(1, &N, howmany, &f[fld][0][0][0], NULL, istride, idist, fft_result, NULL, ostride, odist, FFTW_ESTIMATE);
            fftw_execute(plan);

            for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                    for (k = 0; k < N/2; ++k) {
                        temp = fft_result[(i * N + j) * (N/2+1) + k][0];
                        fft_result[(i * N + j) * (N/2+1) + k][0] = - kIR * k * fft_result[(i * N + j) * (N/2+1) + k][1];
                        fft_result[(i * N + j) * (N/2+1) + k][1] = kIR * k * temp;
                    }
                    fft_result[(i * N + j) * (N/2+1) + N/2][0] = 0.;
                    fft_result[(i * N + j) * (N/2+1) + N/2][1] = 0.;
                }
            }

            plan = fftw_plan_many_dft_c2r(1, &N, howmany, fft_result, NULL, ostride, odist, &f_derivative[fld][0][0][0][0], NULL, istride, idist, FFTW_ESTIMATE);
            fftw_execute(plan);

            for (i = 0; i < N; i++)
                for (j = 0; j < N; ++j)
                    for (k = 0; k < N; ++k) {
                        f_derivative[fld][0][i][j][k] /= N;
                    }


            howmany = N;
            idist = 1, odist = N/2+1;
            istride = N, ostride = 1;

            for (i = 0; i < N; i++) {
                plan = fftw_plan_many_dft_r2c(1, &N, howmany, &f[fld][i][0][0], NULL, istride, idist, fft_result + i * N * (N/2+1), NULL, ostride, odist, FFTW_ESTIMATE);
                fftw_execute(plan);
            }

            for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                    for (k = 0; k < N/2; ++k) {
                        temp = fft_result[(i * N + j) * (N/2+1) + k][0];
                        fft_result[(i * N + j) * (N/2+1) + k][0] = - kIR * k * fft_result[(i * N + j) * (N/2+1) + k][1];
                        fft_result[(i * N + j) * (N/2+1) + k][1] = kIR * k * temp;
                    }
                    fft_result[(i * N + j) * (N/2+1) + N/2][0] = 0.;
                    fft_result[(i * N + j) * (N/2+1) + N/2][1] = 0.;
                }
            }

            for (i = 0; i < N; i++) {
                plan = fftw_plan_many_dft_c2r(1, &N, howmany, fft_result + i * N * (N/2+1), NULL, ostride, odist, &f_derivative[fld][1][i][0][0], NULL, istride, idist, FFTW_ESTIMATE);
                fftw_execute(plan);
            }

            for (i = 0; i < N; i++)
                for (j = 0; j < N; ++j)
                    for (k = 0; k < N; ++k) {
                        f_derivative[fld][1][i][j][k] /= N;
                    }


            howmany = N * N;
            idist = N, odist = N/2+1;
            istride = 1, ostride = 1;

            plan = fftw_plan_many_dft_r2c(1, &N, howmany, &f[fld][0][0][0], NULL, istride, idist, fft_result, NULL, ostride, odist, FFTW_ESTIMATE);
            fftw_execute(plan);


            for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                    for (k = 0; k < N/2; ++k) {
                        temp = fft_result[(i * N + j) * (N/2+1) + k][0];
                        fft_result[(i * N + j) * (N/2+1) + k][0] = - kIR * k * fft_result[(i * N + j) * (N/2+1) + k][1];
                        fft_result[(i * N + j) * (N/2+1) + k][1] = kIR * k * temp;
                    }
                    fft_result[(i * N + j) * (N/2+1) + N/2][0] = 0.;
                    fft_result[(i * N + j) * (N/2+1) + N/2][1] = 0.;
                }
            }

            plan = fftw_plan_many_dft_c2r(1, &N, howmany, fft_result, NULL, ostride, odist, &f_derivative[fld][2][0][0][0], NULL, istride, idist, FFTW_ESTIMATE);
            fftw_execute(plan);

            for (i = 0; i < N; i++)
                for (j = 0; j < N; ++j)
                    for (k = 0; k < N; ++k) {
                        f_derivative[fld][2][i][j][k] /= N;
                    }
            
        }


        fftw_destroy_plan(plan);
        fftw_free(fft_result);

        DerivateIsCalculated = true;
    }

    
    Array3D kernel_GW(const double a, double (&u)[N][N][N], size_t i_GW, size_t j_GW) {
        DECLARE_INDICES
        Array3D kernel_u = {{{0.0}}};

        lapl(u, u_lapl[i_GW][j_GW]);

        if (!DerivateIsCalculated) {
            field_1st_derivative();
        }

        double norm = pow(a, 1 + alpha);
        LOOP {
            kernel_u[i][j][k] = u_lapl[i_GW][j_GW][i][j][k];
            for (size_t fld = 0; fld < nflds; fld++) {
                kernel_u[i][j][k] += 2 * f_derivative[fld][i_GW][i][j][k] * f_derivative[fld][j_GW][i][j][k];
            }
            kernel_u[i][j][k] *= norm;
        }

        return kernel_u;
    }

#endif



    void evolve_VV(double dt) {
        DECLARE_INDICES
        double b, b_half, a_half, a_of_pi;
        int fld;

        Array4D kernel_f;


        #if WITH_GW
        Array3D kernel_u;
        #endif

        isChanged = true;
        b = ad;

        for (int p = 0; p < s; p++) {
            // step 1
            b_half = b + w[p] * dt / 2. * add;

            // step 2
            for (fld = 0; fld < nflds; fld++) {
                kernel_f[fld] = kernel_field(fld);
                LOOP  
                    pi_half[fld][i][j][k] = pi_f[fld][i][j][k] + w[p] * dt / 2. * kernel_f[fld][i][j][k];
            }

            #if WITH_GW
            // step 1 of GW
            for (size_t i_GW = 0; i_GW < 3; i_GW++) {
                for (size_t j_GW = 0; j_GW < 3; j_GW++) {
                    kernel_u = kernel_GW(a, u[i_GW][j_GW], i_GW, j_GW);
                    LOOP {
                        pi_u_half[i_GW][j_GW][i][j][k] = pi_u[i_GW][j_GW][i][j][k] + w[p] * dt / 2 * kernel_u[i][j][k];
                    }
                }
            }
            #endif

            // step 3
            a_half = a + b_half * w[p] * dt / 2.;

            // step 4
            a_of_pi = pow(a_half, alpha - 3);
            for (int fld = 0; fld < nflds; fld++)
                LOOP
                    f[fld][i][j][k] += w[p] * dt * a_of_pi * pi_half[fld][i][j][k];

            #if WITH_GW
            DerivateIsCalculated = false;
            // step 2 of GW
            for (size_t i_GW = 0; i_GW < 3; i_GW++) {
                for (size_t j_GW = 0; j_GW < 3; j_GW++) {
                    LOOP
                        u[i_GW][j_GW][i][j][k] += w[p] * dt * a_of_pi * pi_u_half[i_GW][j_GW][i][j][k];
                }
            }
            #endif


            // step 5
            a = a_half + b_half * w[p] * dt / 2.;

            // step 6
            for (int fld = 0; fld < nflds; fld++) {
                kernel_f[fld] = kernel_field(fld);
                LOOP
                    pi_f[fld][i][j][k] = pi_half[fld][i][j][k] + w[p] * dt / 2 * kernel_f[fld][i][j][k];
            }

            #if WITH_GW
            // step 3 of GW
            for (size_t i_GW = 0; i_GW < 3; i_GW++) {
                for (size_t j_GW = 0; j_GW < 3; j_GW++) {
                    kernel_u = kernel_GW(a, u[i_GW][j_GW], i_GW, j_GW);
                    LOOP
                        pi_u[i_GW][j_GW][i][j][k] = pi_u_half[i_GW][j_GW][i][j][k] + w[p] * dt / 2 * kernel_u[i][j][k];
                }
            }
            #endif

            // step 7
            add = kernel_a();
            b = b_half + w[p] * dt / 2. * add;
        }

        ad = b;
        t += dt;
    }


    #if BIFURCATION
    void evolve_VV_with_perturbation(double dt) {
        DECLARE_INDICES
        double b, b_half, a_half;
        size_t fld;

        double k_mode;
        std::complex<double> dphi_k_half_positive[maxnumbins], dphi_k_half_negative[maxnumbins], dchi_k_half_positive[maxnumbins], dchi_k_half_negative[maxnumbins];
        std::complex<double> pi_k_half_positive[nflds][maxnumbins], pi_k_half_negative[nflds][maxnumbins];
        double phi_av_positive, phi_av_negative, chi_av_positive, chi_av_negative;
        double positive_trajectory[2], negative_trajectory[2];
        double norm1, norm2, norm3;
        Array4D kernel_f;


        #if WITH_GW
        Array3D kernel_u;
        #endif

        isChanged = true;
        b = ad;

        for (size_t p = 0; p < s; p++) {
            // step 1
            b_half = b + w[p] * dt / 2. * add;

            // step 2
            for (fld = 0; fld < nflds; fld++) {
                kernel_f[fld] = kernel_field(fld);
                LOOP  
                    pi_half[fld][i][j][k] = pi_f[fld][i][j][k] + w[p] * dt / 2 * kernel_f[fld][i][j][k];
            }

            #if WITH_GW
            // step 1 of GW
            for (size_t i_GW = 0; i_GW < 3; i_GW++) {
                for (size_t j_GW = 0; j_GW < 3; j_GW++) {
                    kernel_u = kernel_GW(a, u[i_GW][j_GW], i_GW, j_GW);
                    LOOP {
                        pi_u_half[i_GW][j_GW][i][j][k] = pi_u[i_GW][j_GW][i][j][k] + w[p] * dt / 2 * kernel_u[i][j][k];
                    }
                }
            }
            #endif


            // step 1 of perturbation
            LOOP {                
                for (fld = 0; fld < nflds; fld++) {
                    f_positive[fld][i][j][k] = chi_bifurcation_sign[i][j][k] * f[fld][i][j][k];
                    f_negative[fld][i][j][k] = (1 - chi_bifurcation_sign[i][j][k]) * f[fld][i][j][k];
                }
            }

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
                f_av_negative[fld] /= (N3 - count_positive);
            }


            norm1 = pow(a, (1 + alpha));
            norm2 = pow(a, (3 + alpha));

            for (k = 1; k < maxnumbins; k++) {
                k_mode = k * kIR;
                
                for (fld = 0; fld < nflds; fld++) {
                    pi_k_half_positive[fld][k] = 0.;
                    pi_k_half_negative[fld][k] = 0.;

                    for (size_t fld_2 = 0; fld_2 < nflds; fld_2++) {
                        pi_k_half_positive[fld][k] += d2Vdf2(fld + fld_2, f_av_positive) * fk_positive[fld_2][k];
                        pi_k_half_negative[fld][k] += d2Vdf2(fld + fld_2, f_av_negative) * fk_negative[fld_2][k];
                    }

                    pi_k_half_positive[fld][k] *= - norm2;
                    pi_k_half_negative[fld][k] *= - norm2;

                    pi_k_half_positive[fld][k] -= norm1 * pw2(k_mode) * fk_positive[fld][k];
                    pi_k_half_negative[fld][k] -= norm1 * pw2(k_mode) * fk_negative[fld][k];

                    pi_k_half_positive[fld][k] *= w[p] * dt / 2.;
                    pi_k_half_negative[fld][k] *= w[p] * dt / 2.;

                    pi_k_half_positive[fld][k] += pi_k_positive[fld][k];
                    pi_k_half_negative[fld][k] += pi_k_negative[fld][k];
                }
            }


            // step 3
            a_half = a + b_half * w[p] * dt / 2.;

            // step 4
            norm3 = pow(a_half, alpha - 3);
            for (int fld = 0; fld < nflds; fld++)
                LOOP
                    f[fld][i][j][k] += w[p] * dt * norm3 * pi_half[fld][i][j][k];

            
            // step 2 of perturbation
            for (k = 1; k < maxnumbins; k++) {
                for (fld = 0; fld < nflds; fld++) {
                    fk_positive[fld][k] += w[p] * dt * norm3 * pi_k_half_positive[fld][k];
                    fk_negative[fld][k] += w[p] * dt * norm3 * pi_k_half_negative[fld][k];
                }
            }

            #if WITH_GW
            DerivateIsCalculated = false;
            // step 2 of GW
            for (size_t i_GW = 0; i_GW < 3; i_GW++) {
                for (size_t j_GW = 0; j_GW < 3; j_GW++) {
                    LOOP
                        u[i_GW][j_GW][i][j][k] += w[p] * dt * a_of_pi * pi_u_half[i_GW][j_GW][i][j][k];
                }
            }
            #endif


            // step 5
            a = a_half + b_half * w[p] * dt / 2.;

            // step 6
            for (int fld = 0; fld < nflds; fld++) {
                kernel_f[fld] = kernel_field(fld);
                LOOP
                    pi_f[fld][i][j][k] = pi_half[fld][i][j][k] + w[p] * dt / 2 * kernel_f[fld][i][j][k];
            }


            // step 3 of perturbation
            LOOP {                 
                for (fld = 0; fld < nflds; fld++) {
                    f_positive[fld][i][j][k] = chi_bifurcation_sign[i][j][k] * f[fld][i][j][k];
                    f_negative[fld][i][j][k] = (1 - chi_bifurcation_sign[i][j][k]) * f[fld][i][j][k];
                }
            }

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
                f_av_negative[fld] /= (N3 - count_positive);
            }

            for (fld = 0; fld < nflds; fld++) {
                positive_trajectory[fld] = f_av_positive[fld];
                negative_trajectory[fld] = f_av_negative[fld];
            }

            norm1 = pow(a, (1 + alpha));
            norm2 = pow(a, (3 + alpha));

            for (k = 1; k < maxnumbins; k++) {
                k_mode = k * kIR;

                for (fld = 0; fld < nflds; fld++) {
                    pi_k_positive[fld][k] = 0.;
                    pi_k_negative[fld][k] = 0.;

                    for (size_t fld_2 = 0; fld_2 < nflds; fld_2++) {
                        pi_k_positive[fld][k] += d2Vdf2(fld + fld_2, positive_trajectory) * fk_positive[fld_2][k];
                        pi_k_negative[fld][k] += d2Vdf2(fld + fld_2, negative_trajectory) * fk_negative[fld_2][k];
                    }

                    pi_k_positive[fld][k] *= - norm2;
                    pi_k_negative[fld][k] *= - norm2;

                    pi_k_positive[fld][k] -= norm1 * pw2(k_mode) * fk_positive[fld][k];
                    pi_k_negative[fld][k] -= norm1 * pw2(k_mode) * fk_negative[fld][k];

                    pi_k_positive[fld][k] *= w[p] * dt / 2.;
                    pi_k_negative[fld][k] *= w[p] * dt / 2.;

                    pi_k_positive[fld][k] += pi_k_half_positive[fld][k];
                    pi_k_negative[fld][k] += pi_k_half_negative[fld][k];
                }
            }

            #if WITH_GW
            // step 3 of GW
            for (size_t i_GW = 0; i_GW < 3; i_GW++) {
                for (size_t j_GW = 0; j_GW < 3; j_GW++) {
                    kernel_u = kernel_GW(a, u[i_GW][j_GW], i_GW, j_GW);
                    LOOP
                        pi_u[i_GW][j_GW][i][j][k] = pi_u_half[i_GW][j_GW][i][j][k] + w[p] * dt / 2 * kernel_u[i][j][k];
                }
            }
            #endif

            // step 7
            add = kernel_a();
            b = b_half + w[p] * dt / 2. * add;
        }

        ad = b;
        t += dt;
    }
    #endif


#else

    double kinetic_energy_k(size_t fld) {
        DECLARE_INDICES
        double kinetic = 0.;

        #if SIMULATE_INFLATION

        for (k = 1; k < N/2; k++) {
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    if (R_Q2C[fld][i][j][k] < epsilon) {
                        kinetic += pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][1]);
                    }
                }
            }
        }
        kinetic *= 2.;
        for (k = 0; k <= N/2; k += N/2) {
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    if (R_Q2C[fld][i][j][k] < epsilon) {
                        kinetic += pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][1]);
                    }
                }
            }
        }


        #else

        for (k = 1; k < N/2; k++) {
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    kinetic += pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][1]);
                }
            }
        }
        kinetic *= 2.;
        for (k = 0; k <= N/2; k += N/2) {
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    kinetic += pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][0]) + pw2(pi_k[fld][(i * N + j) * (N/2+1) + k][1]);
                }
            }
        }
        #endif

        return(.5 * kinetic / pow(a, 6) / (double)pw2(N3));
    }

    double gradient_energy_k(size_t fld) {
        DECLARE_INDICES
        int px, py, pz;
        double grad = 0.;

        #if SIMULATE_INFLATION

        for (k = 1; k < N/2; k++) {
            pz = k;
            for (i = 0; i < N; i++) {
                px = (i <= N/2) ? i : i - N;
                for (j = 0; j < N; j++) {
                    py = (j <= N/2) ? j : j - N;
                    if (R_Q2C[fld][i][j][k] < epsilon) {
                        grad += pow(kIR, 2) * (px * px + py * py + pz * pz) * (pow(f_k[fld][(i * N + j) * (N/2+1) + k][0], 2) + pow(f_k[fld][(i * N + j) * (N/2+1) + k][1], 2));
                    }
                }
            }
        }
        grad *= 2.;
        for (k = 0; k <= N/2; k+=N/2) {
            pz = k;
            for (i = 0; i < N; i++) {
                px = (i <= N/2) ? i : i - N;
                for (j = 0; j < N; j++) {
                    py = (j <= N/2) ? j : j - N;
                    if (R_Q2C[fld][i][j][k] < epsilon) {
                        grad += pow(kIR, 2) * (px * px + py * py + pz * pz) * (pow(f_k[fld][(i * N + j) * (N/2+1) + k][0], 2) + pow(f_k[fld][(i * N + j) * (N/2+1) + k][1], 2));
                    }
                }
            }
        }

        #else

        for (k = 1; k < N/2; k++) {
            pz = k;
            for (i = 0; i < N; i++) {
                px = (i <= N/2) ? i : i - N;
                for (j = 0; j < N; j++) {
                    py = (j <= N/2) ? j : j - N;
                    grad += pow(kIR, 2) * (px * px + py * py + pz * pz) * (pow(f_k[fld][(i * N + j) * (N/2+1) + k][0], 2) + pow(f_k[fld][(i * N + j) * (N/2+1) + k][1], 2));
                }
            }
        }
        grad *= 2.;
        for (k = 0; k <= N/2; k+=N/2) {
            pz = k;
            for (i = 0; i < N; i++) {
                px = (i <= N/2) ? i : i - N;
                for (j = 0; j < N; j++) {
                    py = (j <= N/2) ? j : j - N;
                    grad += pow(kIR, 2) * (px * px + py * py + pz * pz) * (pow(f_k[fld][(i * N + j) * (N/2+1) + k][0], 2) + pow(f_k[fld][(i * N + j) * (N/2+1) + k][1], 2));
                }
            }
        }

        #endif

        grad /= N3;
        grad /= 2.;

        return(grad / pw2(a) / (double)N3);
    }

    #if SIMULATE_INFLATION
    void Caluculate_R_Q2C(void) {
        DECLARE_INDICES
        size_t fld, k_index;

        double denominator;
        for (fld = 0; fld < nflds; fld++) {
            LOOP_k {
                if (i == 0 && j == 0 && k == 0) continue;
                k_index = (i * N + j) * (N/2+1) + k;

                denominator = sqrt(pw2(f_k[fld][k_index][0]) + pw2(f_k[fld][k_index][1])) * pow(a, 3 - alpha) * sqrt(pw2(ad / a * f_k[fld][k_index][0] + pow(a, alpha - 3) * pi_k[fld][k_index][0]) + pw2(ad / a * f_k[fld][k_index][1] + pow(a, alpha - 3) * pi_k[fld][k_index][1]));
                
                R_Q2C[fld][i][j][k] = DeltaF_DeltaPi / (sqrt(pw2(f_k[fld][k_index][0]) + pw2(f_k[fld][k_index][1])) * pow(a, 3 - alpha) * sqrt(pw2(ad / a * f_k[fld][k_index][0] + pow(a, alpha - 3) * pi_k[fld][k_index][0]) + pw2(ad / a * f_k[fld][k_index][1] + pow(a, alpha - 3) * pi_k[fld][k_index][1])));
                
            }

            R_Q2C[fld][0][0][0] = 0.;
        }
    }
    #endif


    #if !PAD_FLAG
        double kernel_a_k_nopad() {
            DECLARE_INDICES
            double k_a;
            double K_E = 0;
            double G_E = 0;
            double V_E = 0;
            double f_point[nflds];
            size_t fld, term;
            Caluculate_R_Q2C();

            for (fld = 0; fld < nflds; fld++) {
                K_E += kinetic_energy_k(fld);

                G_E += gradient_energy_k(fld);
            }

            LOOP {
                for (fld = 0; fld < nflds; fld++) {
                    f_point[fld] = f[fld][i][j][k];
                }
                for (term = 0; term < num_potential_terms; term++) {
                    V_E += potential_energy(term, f_point);
                }
            }
            V_E /= (double) N3;

            k_a = 1. / 3. * pw2(fStar / MPl) * pow(a, 1 + 2 * alpha) * ((alpha - 2) * K_E + alpha * G_E + (alpha + 1) * V_E);

            return k_a;
        }

        void kernel_field_k_nopad(fftw_complex** kernel_field) {
            DECLARE_INDICES
            size_t fld;
            int k2;
            int px, py, pz;

            double f_point[nflds];
            double norm1 = pow(a, 1 + alpha);
            double norm2 = pow(a, 3 + alpha);


            LOOP {
                for (fld = 0; fld < nflds; fld++)
                    f_point[fld] = f[fld][i][j][k];

                for (fld = 0; fld < nflds; fld++)
                    V_prime[fld][i][j][k] = dvdf(fld, f_point);    
            }

            for (fld = 0; fld < nflds; fld++) {
                plan = fftw_plan_dft_r2c_3d(N, N, N, &V_prime[fld][0][0][0], V_prime_k[fld], FFTW_ESTIMATE);
                fftw_execute(plan);

                for (i = 0; i < N; i++) {
                    px = (i <= N/2 ? i : i - N);
                    for (j = 0; j < N; j++) {
                        py = (j <= N/2 ? j : j - N);
                        for (k = 0; k < N/2+1; k++) {
                            pz = k;
                            k2 = pw2(px) + pw2(py) + pw2(pz);

                            kernel_field[fld][(i * N + j) * (N/2+1) + k][0] = - norm1 * pw2(kIR) * k2 * f_k[fld][(i * N + j) * (N/2+1) + k][0] - norm2 * V_prime_k[fld][(i * N + j) * (N/2+1) + k][0];
                            kernel_field[fld][(i * N + j) * (N/2+1) + k][1] = - norm1 * pw2(kIR) * k2 * f_k[fld][(i * N + j) * (N/2+1) + k][1] - norm2 * V_prime_k[fld][(i * N + j) * (N/2+1) + k][1];
                        }
                    }
                }
            }
        }

        void evolve_VV_k_nopad(double dt) {
            DECLARE_INDICES
            double b, b_half, a_half, norm;
            size_t fld;

            isChanged = true;
            b = ad;

            for (size_t p = 0; p < s; p++) {
                // step 1
                b_half = b + w[p] * dt / 2. * add;

                // step 2
                for (fld = 0; fld < nflds; fld++) {
                    std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
                    plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
                    fftw_execute(plan_f);

                    LOOP
                        f[fld][i][j][k] /= N3;
                }
                kernel_field_k_nopad(kernel_f_k);
                for (fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        pi_k_half[fld][(i * N + j) * (N/2+1) + k][0] = pi_k[fld][(i * N + j) * (N/2+1) + k][0] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][0];
                        pi_k_half[fld][(i * N + j) * (N/2+1) + k][1] = pi_k[fld][(i * N + j) * (N/2+1) + k][1] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][1];
                    }
                }

                // step 3
                a_half = a + b_half * w[p] * dt / 2.;

                // step 4
                norm = pow(a_half, alpha - 3);
                for (fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        f_k[fld][(i * N + j) * (N/2+1) + k][0] += w[p] * dt * norm * pi_k_half[fld][(i * N + j) * (N/2+1) + k][0];
                        f_k[fld][(i * N + j) * (N/2+1) + k][1] += w[p] * dt * norm * pi_k_half[fld][(i * N + j) * (N/2+1) + k][1];
                    }

                    std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
                    plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
                    fftw_execute(plan_f);

                    LOOP
                        f[fld][i][j][k] /= N3;
                }

                // step 5
                a = a_half + b_half * w[p] * dt / 2.;

                // step 6
                kernel_field_k_nopad(kernel_f_k);
                for (fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        pi_k[fld][(i * N + j) * (N/2+1) + k][0] = pi_k_half[fld][(i * N + j) * (N/2+1) + k][0] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][0];
                        pi_k[fld][(i * N + j) * (N/2+1) + k][1] = pi_k_half[fld][(i * N + j) * (N/2+1) + k][1] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][1];
                    }
                }

                // step 7
                add = kernel_a_k_nopad();
                b = b_half + w[p] * dt / 2. * add;
            }

            ad = b;
            t += dt;
        }

        void evolve_RK45_k_nopad(double dt) {
            DECLARE_INDICES
            double b, b_half, a_half, norm;
            size_t fld;

            isChanged = true;
            b = ad;

            for (size_t p = 0; p < s; p++) {
                // step 1
                b_half = b + w[p] * dt / 2. * add;

                // step 2
                for (fld = 0; fld < nflds; fld++) {
                    std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
                    plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
                    fftw_execute(plan_f);

                    LOOP
                        f[fld][i][j][k] /= N3;
                }
                kernel_field_k_nopad(kernel_f_k);
                for (fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        pi_k_half[fld][(i * N + j) * (N/2+1) + k][0] = pi_k[fld][(i * N + j) * (N/2+1) + k][0] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][0];
                        pi_k_half[fld][(i * N + j) * (N/2+1) + k][1] = pi_k[fld][(i * N + j) * (N/2+1) + k][1] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][1];
                    }
                }

                // step 3
                a_half = a + b_half * w[p] * dt / 2.;

                // step 4
                norm = pow(a_half, alpha - 3);
                for (fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        f_k[fld][(i * N + j) * (N/2+1) + k][0] += w[p] * dt * norm * pi_k_half[fld][(i * N + j) * (N/2+1) + k][0];
                        f_k[fld][(i * N + j) * (N/2+1) + k][1] += w[p] * dt * norm * pi_k_half[fld][(i * N + j) * (N/2+1) + k][1];
                    }

                    std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
                    plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
                    fftw_execute(plan_f);

                    LOOP
                        f[fld][i][j][k] /= N3;
                }

                // step 5
                a = a_half + b_half * w[p] * dt / 2.;

                // step 6
                kernel_field_k_nopad(kernel_f_k);
                for (fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        pi_k[fld][(i * N + j) * (N/2+1) + k][0] = pi_k_half[fld][(i * N + j) * (N/2+1) + k][0] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][0];
                        pi_k[fld][(i * N + j) * (N/2+1) + k][1] = pi_k_half[fld][(i * N + j) * (N/2+1) + k][1] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][1];
                    }
                }

                // step 7
                add = kernel_a_k_nopad();
                b = b_half + w[p] * dt / 2. * add;
            }

            ad = b;
            t += dt;
        }

    #else

        void padding(fftw_complex* f_k, fftw_complex* f_k_pad) {
            int i, j, k, i_pad, j_pad;

            for (k = 0; k < N/2; k++) {
                f_k_pad[(0 * N_pad + 0) * (N_pad/2+1) + k][0] = f_k[(0 * N + 0) * (N/2+1) + k][0];
                f_k_pad[(0 * N_pad + 0) * (N_pad/2+1) + k][1] = f_k[(0 * N + 0) * (N/2+1) + k][1];
                for (j = 1; j <= N/2; j++) {
                    j_pad = N_pad - j;
                    for (i = 1; i <= N/2; i++) {
                        i_pad = N_pad - i;

                        f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] = f_k[(i * N + j) * (N/2+1) + k][0];
                        f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] = f_k[(i * N + j) * (N/2+1) + k][1];

                        f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] = f_k[((N-i) * N + j) * (N/2+1) + k][0];
                        f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] = f_k[((N-i) * N + j) * (N/2+1) + k][1];

                        f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] = f_k[(i * N + (N-j)) * (N/2+1) + k][0];
                        f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] = f_k[(i * N + (N-j)) * (N/2+1) + k][1];

                        f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][0] = f_k[((N-i) * N + (N-j)) * (N/2+1) + k][0];
                        f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][1] = f_k[((N-i) * N + (N-j)) * (N/2+1) + k][1];
                    }
                }
                j = 0;
                for (i = 1; i <= N/2; i++) {
                    i_pad = N_pad - i;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] = f_k[(i * N + j) * (N/2+1) + k][0];
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] = f_k[(i * N + j) * (N/2+1) + k][1];

                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] = f_k[((N-i) * N + j) * (N/2+1) + k][0];
                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] = f_k[((N-i) * N + j) * (N/2+1) + k][1];
                }
                i = 0;
                for (j = 1; j <= N/2; j++) {
                    j_pad = N_pad - j;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] = f_k[(i * N + j) * (N/2+1) + k][0];
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] = f_k[(i * N + j) * (N/2+1) + k][1];

                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] = f_k[(i * N + (N-j)) * (N/2+1) + k][0];
                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] = f_k[(i * N + (N-j)) * (N/2+1) + k][1];
                }
                i = N / 2;
                i_pad = N_pad - i;
                for (j = 1; j <= N/2; j++) {
                    j_pad = N_pad - j;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;
                }
                j = N / 2;
                j_pad = N_pad - j;
                for (i = 1; i <= N/2; i++) {
                    i_pad = N_pad - i;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;
                }
                j = 0;
                i = N/2;
                    i_pad = N_pad - i;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                i = 0;
                j = N/2;
                    j_pad = N_pad - j;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;
            }

                k = N/2;
                f_k_pad[(0 * N_pad + 0) * (N_pad/2+1) + k][0] = f_k[(0 * N + 0) * (N/2+1) + k][0] / 2;
                f_k_pad[(0 * N_pad + 0) * (N_pad/2+1) + k][1] = f_k[(0 * N + 0) * (N/2+1) + k][1] / 2;
                for (j = 1; j <= N/2; j++) {
                    j_pad = N_pad - j;
                    for (i = 1; i <= N/2; i++) {
                        i_pad = N_pad - i;

                        f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] = f_k[(i * N + j) * (N/2+1) + k][0] / 2;
                        f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] = f_k[(i * N + j) * (N/2+1) + k][1] / 2;

                        f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] = f_k[((N-i) * N + j) * (N/2+1) + k][0] / 2;
                        f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] = f_k[((N-i) * N + j) * (N/2+1) + k][1] / 2;

                        f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] = f_k[(i * N + (N-j)) * (N/2+1) + k][0] / 2;
                        f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] = f_k[(i * N + (N-j)) * (N/2+1) + k][1] / 2;

                        f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][0] = f_k[((N-i) * N + (N-j)) * (N/2+1) + k][0] / 2;
                        f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][1] = f_k[((N-i) * N + (N-j)) * (N/2+1) + k][1] / 2;
                    }
                }
                j = 0;
                for (i = 1; i <= N/2; i++) {
                    i_pad = N_pad - i;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] = f_k[(i * N + j) * (N/2+1) + k][0] / 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] = f_k[(i * N + j) * (N/2+1) + k][1] / 2;

                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] = f_k[((N-i) * N + j) * (N/2+1) + k][0] / 2;
                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] = f_k[((N-i) * N + j) * (N/2+1) + k][1] / 2;
                }
                i = 0;
                for (j = 1; j <= N/2; j++) {
                    j_pad = N_pad - j;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] = f_k[(i * N + j) * (N/2+1) + k][0] / 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] = f_k[(i * N + j) * (N/2+1) + k][1] / 2;

                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] = f_k[(i * N + (N-j)) * (N/2+1) + k][0] / 2;
                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] = f_k[(i * N + (N-j)) * (N/2+1) + k][1] / 2;
                }
                i = N / 2;
                i_pad = N_pad - i;
                for (j = 1; j <= N/2; j++) {
                    j_pad = N_pad - j;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;
                }
                j = N / 2;
                j_pad = N_pad - j;
                for (i = 1; i <= N/2; i++) {
                    i_pad = N_pad - i;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;
                }
                j = 0;
                i = N/2;
                    i_pad = N_pad - i;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                i = 0;
                j = N/2;
                    j_pad = N_pad - j;

                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] /= 2;

                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] /= 2;
                    f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] /= 2;

        }


        void extract_padding(fftw_complex* f_k_pad, fftw_complex* f_k_extract) {
            int i, j, k, i_pad, j_pad;
            for (k = 0; k < N/2; k++) {
                for (j = 1; j < N/2; j++) {
                    j_pad = N_pad - j;
                    for (i = 1; i < N/2; i++) {
                        i_pad = N_pad - i;

                        f_k_extract[(i * N + j) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0];
                        f_k_extract[(i * N + j) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1];

                        f_k_extract[((N-i) * N + j) * (N/2+1) + k][0] = f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0];
                        f_k_extract[((N-i) * N + j) * (N/2+1) + k][1] = f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1];

                        f_k_extract[(i * N + (N-j)) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0];
                        f_k_extract[(i * N + (N-j)) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1];

                        f_k_extract[((N-i) * N + (N-j)) * (N/2+1) + k][0] = f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][0];
                        f_k_extract[((N-i) * N + (N-j)) * (N/2+1) + k][1] = f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][1];
                    }
                }
                for (j = 0; j <= N/2; j += N/2) {
                    for (i = 1; i < N/2; i++) {
                        i_pad = N_pad - i;

                        f_k_extract[(i * N + j) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0];
                        f_k_extract[(i * N + j) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1];

                        f_k_extract[((N-i) * N + j) * (N/2+1) + k][0] = f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0];
                        f_k_extract[((N-i) * N + j) * (N/2+1) + k][1] = f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1];
                    }
                }

                for (i = 0; i <= N/2; i += N/2) {
                    for (j = 1; j < N/2; j++) {
                        j_pad = N_pad - j;

                        f_k_extract[(i * N + j) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0];
                        f_k_extract[(i * N + j) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1];

                        f_k_extract[(i * N + (N-j)) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0];
                        f_k_extract[(i * N + (N-j)) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1];
                    }
                }

                for (i = 0; i <= N/2; i += N/2) {
                    for (j = 0; j <= N/2; j += N/2) {
                        j_pad = N_pad - j;

                        f_k_extract[(i * N + j) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0];
                        f_k_extract[(i * N + j) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1];
                    }
                }

                j = N/2;
                for (i = 0; i < N; i++) {
                    f_k_extract[(i * N + j) * (N/2+1) + k][0] *= 2;
                    f_k_extract[(i * N + j) * (N/2+1) + k][1] *= 2;
                }

                i = N/2;
                for (j = 0; j < N; j++) {
                    f_k_extract[(i * N + j) * (N/2+1) + k][0] *= 2;
                    f_k_extract[(i * N + j) * (N/2+1) + k][1] *= 2;
                }
            }


            k = N/2;
            for (j = 1; j < N/2; j++) {
                j_pad = N_pad - j;
                for (i = 1; i < N/2; i++) {
                    i_pad = N_pad - i;

                    f_k_extract[(i * N + j) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[(i * N + j) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] * 2;

                    f_k_extract[((N-i) * N + j) * (N/2+1) + k][0] = f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[((N-i) * N + j) * (N/2+1) + k][1] = f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] * 2;

                    f_k_extract[(i * N + (N-j)) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[(i * N + (N-j)) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] * 2;

                    f_k_extract[((N-i) * N + (N-j)) * (N/2+1) + k][0] = f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[((N-i) * N + (N-j)) * (N/2+1) + k][1] = f_k_pad[(i_pad * N_pad + j_pad) * (N_pad/2+1) + k][1] * 2;
                }
            }

            for (j = 0; j <= N/2; j += N/2) {
                for (i = 1; i < N/2; i++) {
                    i_pad = N_pad - i;

                    f_k_extract[(i * N + j) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[(i * N + j) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] * 2;

                    f_k_extract[((N-i) * N + j) * (N/2+1) + k][0] = f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[((N-i) * N + j) * (N/2+1) + k][1] = f_k_pad[(i_pad * N_pad + j) * (N_pad/2+1) + k][1] * 2;
                }
            }

            for (i = 0; i <= N/2; i += N/2) {
                for (j = 1; j < N/2; j++) {
                    j_pad = N_pad - j;

                    f_k_extract[(i * N + j) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[(i * N + j) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] * 2;

                    f_k_extract[(i * N + (N-j)) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[(i * N + (N-j)) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j_pad) * (N_pad/2+1) + k][1] * 2;
                }
            }

            for (i = 0; i <= N/2; i += N/2) {
                for (j = 0; j <= N/2; j += N/2) {
                    j_pad = N_pad - j;

                    f_k_extract[(i * N + j) * (N/2+1) + k][0] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][0] * 2;
                    f_k_extract[(i * N + j) * (N/2+1) + k][1] = f_k_pad[(i * N_pad + j) * (N_pad/2+1) + k][1] * 2;
                }
            }

            j = N/2;
            for (i = 0; i < N; i++) {
                f_k_extract[(i * N + j) * (N/2+1) + k][0] *= 2;
                f_k_extract[(i * N + j) * (N/2+1) + k][1] *= 2;
            }

            i = N/2;
            for (j = 0; j < N; j++) {
                f_k_extract[(i * N + j) * (N/2+1) + k][0] *= 2;
                f_k_extract[(i * N + j) * (N/2+1) + k][1] *= 2;
            }


            double normalize_factor = pow((double) N_pad / N, 3);
            for (k = 0; k < N/2+1; k++) {
                for (i = 0; i < N; i++) {
                    for (j = 0; j < N; j++) {
                        f_k_extract[(i * N + j) * (N/2+1) + k][0] /= normalize_factor;
                        f_k_extract[(i * N + j) * (N/2+1) + k][1] /= normalize_factor;
                    }
                }
            }
        }
        double kernel_a_k_pad(const double a, double (&f)[nflds][N][N][N], double (&f_pad)[nflds][N_pad][N_pad][N_pad], fftw_complex** pi_k) {
            DECLARE_INDICES
            double k_a;
            double K_E = 0;
            double G_E = 0;
            double V_E = 0;
            double f_point[nflds];
            size_t fld;

            for (fld = 0; fld < nflds; fld++) {
                K_E += kinetic_energy_k(fld);

                G_E += gradient_energy_k(fld);
            }

            LOOP_pad {
                for (fld = 0; fld < nflds; fld++) {
                    f_point[fld] = f_pad[fld][i][j][k];
                }
                for (int term = 0; term < num_potential_terms; term++) {
                    V_E += potential_energy(term, f_point);
                }
            }
            V_E /= (double) N3_pad;

            k_a = 1. / 3. * pw2(fStar / MPl) * pow(a, 1 + 2 * alpha) * ((alpha - 2) * K_E + alpha * G_E + (alpha + 1) * V_E);
            return k_a;
        }

        void kernel_field_k_pad(const double a, double (&f_pad)[nflds][N_pad][N_pad][N_pad], fftw_complex** f_k, fftw_complex** kernel_field) {
            DECLARE_INDICES
            int fld, k2;
            int px, py, pz;

            double f_point[nflds];
            double norm1 = pow(a, 1 + alpha);
            double norm2 = pow(a, 3 + alpha);


            LOOP_pad {
                for (fld = 0; fld < nflds; fld++)
                    f_point[fld] = f_pad[fld][i][j][k];

                for (fld = 0; fld < nflds; fld++)
                    V_prime_pad[fld][i][j][k] = dvdf(fld, f_point);    
            }

            for (fld = 0; fld < nflds; fld++) {
                plan_pad = fftw_plan_dft_r2c_3d(N_pad, N_pad, N_pad, &V_prime_pad[fld][0][0][0], V_prime_k_pad[fld], FFTW_ESTIMATE);
                fftw_execute(plan_pad);

                extract_padding(V_prime_k_pad[fld], V_prime_k[fld]);

                for (i = 0; i < N; i++) {
                    px = (i <= N/2 ? i : i - N);
                    for (j = 0; j < N; j++) {
                        py = (j <= N/2 ? j : j - N);
                        for (k = 0; k < N/2+1; k++) {
                            pz = k;
                            k2 = pw2(px) + pw2(py) + pw2(pz);

                            kernel_field[fld][(i * N + j) * (N/2+1) + k][0] = - norm1 * pw2(kIR) * k2 * f_k[fld][(i * N + j) * (N/2+1) + k][0] - norm2 * V_prime_k[fld][(i * N + j) * (N/2+1) + k][0];
                            kernel_field[fld][(i * N + j) * (N/2+1) + k][1] = - norm1 * pw2(kIR) * k2 * f_k[fld][(i * N + j) * (N/2+1) + k][1] - norm2 * V_prime_k[fld][(i * N + j) * (N/2+1) + k][1];
                        }
                    }
                }
            }
        }

        void evolve_VV_k_pad(double dt) {
            DECLARE_INDICES
            double b, b_half, a_half, a_of_pi;
            int fld;

            isChanged = true;
            b = ad;

            for (int p = 0; p < s; p++) {
                // step 1
                b_half = b + w[p] * dt / 2. * add;

                // step 2
                for (int fld = 0; fld < nflds; fld++) {
                    memset(f_k_pad[fld], 0, sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
                    padding(f_k[fld], f_k_pad[fld]);
                    std::memcpy(Cache_Data_k_pad, f_k_pad[fld], sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
                    plan_f = fftw_plan_dft_c2r_3d(N_pad, N_pad, N_pad, Cache_Data_k_pad, &f_pad[fld][0][0][0], FFTW_ESTIMATE);
                    fftw_execute(plan_f);

                    LOOP_pad
                        f_pad[fld][i][j][k] /= N3;
                }
                kernel_field_k_pad(a, f_pad, f_k, kernel_f_k);
                for (fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        pi_k_half[fld][(i * N + j) * (N/2+1) + k][0] = pi_k[fld][(i * N + j) * (N/2+1) + k][0] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][0];
                        pi_k_half[fld][(i * N + j) * (N/2+1) + k][1] = pi_k[fld][(i * N + j) * (N/2+1) + k][1] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][1];
                    }
                }

                // step 3
                a_half = a + b_half * w[p] * dt / 2.;

                // step 4
                a_of_pi = pow(a_half, -(3 - alpha));
                for (int fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        f_k[fld][(i * N + j) * (N/2+1) + k][0] += w[p] * dt * a_of_pi * pi_k_half[fld][(i * N + j) * (N/2+1) + k][0];
                        f_k[fld][(i * N + j) * (N/2+1) + k][1] += w[p] * dt * a_of_pi * pi_k_half[fld][(i * N + j) * (N/2+1) + k][1];
                    }

                    std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
                    plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
                    fftw_execute(plan_f);

                    LOOP
                        f[fld][i][j][k] /= N3;

                    memset(f_k_pad[fld], 0, sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
                    padding(f_k[fld], f_k_pad[fld]);
                    std::memcpy(Cache_Data_k_pad, f_k_pad[fld], sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
                    plan_f = fftw_plan_dft_c2r_3d(N_pad, N_pad, N_pad, Cache_Data_k_pad, &f_pad[fld][0][0][0], FFTW_ESTIMATE);
                    fftw_execute(plan_f);

                    LOOP_pad
                        f_pad[fld][i][j][k] /= N3;
                }

                // step 5
                a = a_half + b_half * w[p] * dt / 2.;

                // step 6
                kernel_field_k_pad(a, f_pad, f_k, kernel_f_k);
                for (int fld = 0; fld < nflds; fld++) {
                    LOOP_k {
                        pi_k[fld][(i * N + j) * (N/2+1) + k][0] = pi_k_half[fld][(i * N + j) * (N/2+1) + k][0] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][0];
                        pi_k[fld][(i * N + j) * (N/2+1) + k][1] = pi_k_half[fld][(i * N + j) * (N/2+1) + k][1] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][1];
                    }
                }

                // step 7
                add = kernel_a_k_pad(a, f, f_pad, pi_k);
                b = b_half + w[p] * dt / 2. * add;
            }

            ad = b;
            t += dt;
        }
    #endif
#endif






#if BIFURCATION & SPECTRAL_FLAG
void evolve_VV_k_nopad_with_perturbation(double dt) {
    DECLARE_INDICES
    double b, b_half, a_half, a_of_pi;
    size_t fld;
    double k_mode;
    std::complex<double> pi_k_half_positive[nflds][maxnumbins], pi_k_half_negative[nflds][maxnumbins];


    double phi_av_positive, phi_av_negative, chi_av_positive, chi_av_negative;
    double positive_trajectory[2], negative_trajectory[2];
    double norm1, norm2, norm3;
    isChanged = true;
    b = ad;

    for (size_t p = 0; p < s; p++) {
        // step 1
        b_half = b + w[p] * dt / 2. * add;

        // step 2
        for (fld = 0; fld < nflds; fld++) {
            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(plan_f);

            LOOP
                f[fld][i][j][k] /= N3;
        }
        kernel_field_k_nopad(kernel_f_k);
        for (fld = 0; fld < nflds; fld++) {
            LOOP_k {
                pi_k_half[fld][(i * N + j) * (N/2+1) + k][0] = pi_k[fld][(i * N + j) * (N/2+1) + k][0] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][0];
                pi_k_half[fld][(i * N + j) * (N/2+1) + k][1] = pi_k[fld][(i * N + j) * (N/2+1) + k][1] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][1];
            }
        }

        // step 1 of perturbation
        LOOP {                
            for (fld = 0; fld < nflds; fld++) {
                f_positive[fld][i][j][k] = chi_bifurcation_sign[i][j][k] * f[fld][i][j][k];
                f_negative[fld][i][j][k] = (1 - chi_bifurcation_sign[i][j][k]) * f[fld][i][j][k];
            }
        }

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
            f_av_negative[fld] /= (N3 - count_positive);
        }

        for (fld = 0; fld < nflds; fld++) {
            positive_trajectory[fld] = f_av_positive[fld];
            negative_trajectory[fld] = f_av_negative[fld];
        }

        norm1 = pow(a, (1 + alpha));
        norm2 = pow(a, (3 + alpha));

        for (k = 1; k < maxnumbins; k++) {
            k_mode = k * kIR;
            
            for (fld = 0; fld < nflds; fld++) {
                pi_k_half_positive[fld][k] = 0.;
                pi_k_half_negative[fld][k] = 0.;

                for (size_t fld_2 = 0; fld_2 < nflds; fld_2++) {
                    pi_k_half_positive[fld][k] += d2Vdf2(fld + fld_2, positive_trajectory) * fk_positive[fld_2][k];
                    pi_k_half_negative[fld][k] += d2Vdf2(fld + fld_2, negative_trajectory) * fk_negative[fld_2][k];
                }

                pi_k_half_positive[fld][k] *= - norm2;
                pi_k_half_negative[fld][k] *= - norm2;

                pi_k_half_positive[fld][k] -= norm1 * pw2(k_mode) * fk_positive[fld][k];
                pi_k_half_negative[fld][k] -= norm1 * pw2(k_mode) * fk_negative[fld][k];

                pi_k_half_positive[fld][k] *= w[p] * dt / 2.;
                pi_k_half_negative[fld][k] *= w[p] * dt / 2.;

                pi_k_half_positive[fld][k] += pi_k_positive[fld][k];
                pi_k_half_negative[fld][k] += pi_k_negative[fld][k];

            }
        }

        // step 3
        a_half = a + b_half * w[p] * dt / 2.;

        // step 4
        norm3 = pow(a_half, alpha - 3);
        for (fld = 0; fld < nflds; fld++) {
            LOOP_k {
                f_k[fld][(i * N + j) * (N/2+1) + k][0] += w[p] * dt * norm3 * pi_k_half[fld][(i * N + j) * (N/2+1) + k][0];
                f_k[fld][(i * N + j) * (N/2+1) + k][1] += w[p] * dt * norm3 * pi_k_half[fld][(i * N + j) * (N/2+1) + k][1];
            }

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(plan_f);

            LOOP
                f[fld][i][j][k] /= N3;
        }

        // step 2 of perturbation
        // norm3 = pow(a_half, -(3 - alpha));
        for (k = 1; k < maxnumbins; k++) {
            for (fld = 0; fld < nflds; fld++) {
                fk_positive[fld][k] += w[p] * dt * norm3 * pi_k_half_positive[fld][k];
                fk_negative[fld][k] += w[p] * dt * norm3 * pi_k_half_negative[fld][k];
            }
        }

        // step 5
        a = a_half + b_half * w[p] * dt / 2.;

        // step 6
        kernel_field_k_nopad(kernel_f_k);
        for (fld = 0; fld < nflds; fld++) {
            LOOP_k {
                pi_k[fld][(i * N + j) * (N/2+1) + k][0] = pi_k_half[fld][(i * N + j) * (N/2+1) + k][0] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][0];
                pi_k[fld][(i * N + j) * (N/2+1) + k][1] = pi_k_half[fld][(i * N + j) * (N/2+1) + k][1] + w[p] * dt / 2 * kernel_f_k[fld][(i * N + j) * (N/2+1) + k][1];
            }
        }

        // step 3 of perturbation
        LOOP {                 
            for (fld = 0; fld < nflds; fld++) {
                f_positive[fld][i][j][k] = chi_bifurcation_sign[i][j][k] * f[fld][i][j][k];
                f_negative[fld][i][j][k] = (1 - chi_bifurcation_sign[i][j][k]) * f[fld][i][j][k];
            }
        }

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
            f_av_negative[fld] /= (N3 - count_positive);
        }

        for (fld = 0; fld < nflds; fld++) {
            positive_trajectory[fld] = f_av_positive[fld];
            negative_trajectory[fld] = f_av_negative[fld];
        }

        norm1 = pow(a, (1 + alpha));
        norm2 = pow(a, (3 + alpha));

        for (k = 1; k < maxnumbins; k++) {
            k_mode = k * kIR;

            for (fld = 0; fld < nflds; fld++) {
                pi_k_positive[fld][k] = 0.;
                pi_k_negative[fld][k] = 0.;

                for (size_t fld_2 = 0; fld_2 < nflds; fld_2++) {
                    pi_k_positive[fld][k] += d2Vdf2(fld + fld_2, positive_trajectory) * fk_positive[fld_2][k];
                    pi_k_negative[fld][k] += d2Vdf2(fld + fld_2, negative_trajectory) * fk_negative[fld_2][k];
                
                }

                pi_k_positive[fld][k] *= - norm2;
                pi_k_negative[fld][k] *= - norm2;

                pi_k_positive[fld][k] -= norm1 * pw2(k_mode) * fk_positive[fld][k];
                pi_k_negative[fld][k] -= norm1 * pw2(k_mode) * fk_negative[fld][k];

                pi_k_positive[fld][k] *= w[p] * dt / 2.;
                pi_k_negative[fld][k] *= w[p] * dt / 2.;

                pi_k_positive[fld][k] += pi_k_half_positive[fld][k];
                pi_k_negative[fld][k] += pi_k_half_negative[fld][k];

            }
        }
        
        // step 7
        add = kernel_a_k_nopad();
        b = b_half + w[p] * dt / 2. * add;
    }

    ad = b;
    t += dt;
}
#endif


#if BIFURCATION

void bifurcation_sign() {
    DECLARE_INDICES
    size_t fld;

    #if SPECTRAL_FLAG
    // if (isChanged & useSpectral) {
    if (isChanged) {
        double a_of_fd = pow(a, - (3 - alpha));
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

    count_positive = 0;

    LOOP {
        if (f[1][i][j][k] > 0) {
            chi_bifurcation_sign[i][j][k] = 1;
            count_positive++;
        } else {
            chi_bifurcation_sign[i][j][k] = 0;
        }
    }
    std::cout << std::endl << "count_positive: " << count_positive << std::endl;

}


void bifurcation() {
    DECLARE_INDICES
    int fld;

    #if SPECTRAL_FLAG
    // if (isChanged & useSpectral) {
    if (isChanged) {
        double a_of_fd = pow(a, - (3 - alpha));
        for (fld = 0; fld < nflds; fld++) {
            fftw_plan forward_plan_f;
            fftw_plan forward_plan_pi;
            // fftw_plan forward_plan_fd;

            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            forward_plan_pi = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(forward_plan_pi);

            // std::memcpy(Cache_Data_k, fd_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            // forward_plan_fd = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &fd[fld][0][0][0], FFTW_ESTIMATE);
            // fftw_execute(forward_plan_fd);

            LOOP {
                f[fld][i][j][k] /= N3;
                pi_f[fld][i][j][k] /= N3;
                // fd[fld][i][j][k] = a_of_fd * pi_f[fld][i][j][k];
            }
            fftw_destroy_plan(forward_plan_f);
            fftw_destroy_plan(forward_plan_pi);
            // fftw_destroy_plan(forward_plan_fd);
        }
        isChanged = false;
    }
    #endif

    // count_positive = 0;

    LOOP {

        for (fld = 0; fld < nflds; fld++) {
            f_positive[fld][i][j][k] = chi_bifurcation_sign[i][j][k] * f[fld][i][j][k];
            f_negative[fld][i][j][k] = (1 - chi_bifurcation_sign[i][j][k]) * f[fld][i][j][k];
        }
    }

}

#endif
