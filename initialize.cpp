// Copyright (c) 2024 Jie Jiang (江捷) <jiejiang@pusan.ac.kr>
// This code is licensed under the MIT License.
// See the LICENSE file in the project root for license information.

#include "pspectcosmo.h"

// extern double dt;

#define randa 16807
#define randm 2147483647
#define randq 127773
#define randr 2836
double rand_uniform(void)
{
    if (seed < 1) return(0.33); // *DEBUG* This is used to avoid randomness, for debugging purposes only.
    static int i = 0;
    static int next = seed;
    if (!(next > 0)) { // Guard against 0, negative, or other invalid seeds
        std::cout << "Invalid seed used in random number function. Using seed = 1" << std::endl;
        next = 1;
    }
    if (i == 0) // On the first call run through 100 calls. This allows small seeds without the first results necessarily being small.
    for (i = 1; i < 100; i++)
        rand_uniform();
    next = randa * (next % randq) - randr * (next / randq);
    if (next < 0) next += randm;
    return ((double)next / (double)randm);
}
#undef randa
#undef randm
#undef randq
#undef randr


// 计算向量的模长
double norm(const Vector3 &v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// 归一化向量
Vector3 normalize(const Vector3 &v) {
    double v_norm = norm(v);
    if (v_norm == 0) {
        throw std::invalid_argument("Vector magnitude cannot be zero.");
    }
    return {v[0] / v_norm, v[1] / v_norm, v[2] / v_norm};
}


// 计算向量的叉积
Vector3 cross(const Vector3 &a, const Vector3 &b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// 检查向量是否接近另一个向量
bool is_approx(const Vector3 &a, const Vector3 &b, double tol = 1e-9) {
    return std::abs(a[0] - b[0]) < tol && std::abs(a[1] - b[1]) < tol && std::abs(a[2] - b[2]) < tol;
}


// 函数定义
std::pair<Vector3, Vector3> orthogonal_unit_vectors(const Vector3 &k) {
    double k_norm = norm(k);
    if (k_norm == 0) {
        throw std::invalid_argument("The magnitude of the momentum vector k cannot be zero.");
    }

    Vector3 k_hat = normalize(k);

    // 选择一个与 k_hat 不平行的向量
    Vector3 temp;
    if (is_approx(k_hat, {0, 0, 1})) {
        temp = {0, 1, 0};
    } else {
        temp = {0, 0, 1};
    }

    Vector3 u = normalize(cross(k_hat, temp));
    Vector3 v = normalize(cross(k_hat, u));

    // 确保 u 和 v 的分量为正
    for (int i = 0; i < 3; ++i) {
        if (u[i] < 0) {
            for (auto &component : u) {
                component = -component;
            }
            break;
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (v[i] < 0) {
            for (auto &component : v) {
                component = -component;
            }
            break;
        }
    }

    return std::make_pair(u, v);
}


// 计算极化张量 e_plus 和 e_cross
Matrix3x3 TT_tensors(const Vector3 &k) {
    auto [u, v] = orthogonal_unit_vectors(k);

    Matrix3x3 e_TT {};

    // 计算 e_cross = u * v^T + v * u^T
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            e_TT[i][j] = u[i] * u[j] - v[i] * v[j] + u[i] * v[j] + v[i] * u[j];
        }
    }

    return e_TT;
}





void set_mode(double p2, double m2, fftw_complex *f_k, fftw_complex *pi_k, bool isReal) {
    double phase, amplitude_l, amplitude_r, rms_amplitude, omega;
    double re_f_left, im_f_left, re_f_right, im_f_right;

    // static double norm = omegaStar / fStar * pow(N / dx, 1.5) / sqrt(2.);
    // static double norm = omegaStar / fStar * pow(N / dx, 0.5) / sqrt(2.);
    static double norm = sqrt(DeltaF_DeltaPi);

    static int tachyonic = 0; // Avoid printing the same error repeatedly

    // Momentum cutoff. If kcutoff!=0 then eliminate all initial modes with k>kcutoff.
    static double k2cutoff = (kcutoff < 2. * pi * (double)N / L ? pw2(kcutoff) : 0.);
    if (k2cutoff > 0. && p2 > k2cutoff) {
        f_k[0][0] = 0.;
        f_k[0][1] = 0.;
        pi_k[0][0] = 0.;
        pi_k[0][1] = 0.;
        return;
    }

    if (p2 + m2 > 0.) // Check to avoid doubleing point errors
        omega = sqrt(p2 + pw2(a) * m2); // Omega = Sqrt(p^2 + m^2)
    else {
        if (tachyonic == 0)
            std::cout << "Warning: Tachyonic mode(s) may be initialized inaccurately" << std::endl;
        omega = sqrt(p2); // If p^2 + m^2 < 0 use m^2=0
        tachyonic = 1;
    }

    if (omega > 0.) // Avoid dividing by zero
        rms_amplitude = norm / sqrt(omega);// * pow(p2, .75 - (double)NDIMS / 4.);
    else
        rms_amplitude = 0.;



    #if SIMULATE_INFLATION

    amplitude_l = rms_amplitude / a;
    amplitude_r = 0;

    // Left moving component
    phase = 2. * pi * rand_uniform(); // Set phase randomly
    re_f_left = amplitude_l * cos(phase);
    im_f_left = amplitude_l * sin(phase);

    f_k[0][0] = re_f_left; // Re(field)
    f_k[0][1] = im_f_left; // Im(field)
    pi_k[0][0] = pow(a, alpha - 3) * (-omega * im_f_left - hubble_init * f_k[0][0]); // Field derivative
    pi_k[0][1] = pow(a, alpha - 3) * (omega * re_f_left - hubble_init * f_k[0][1]);

    #else

    amplitude_l = rms_amplitude / sqrt(2.) * sqrt(log(1. / rand_uniform())) / a;
    amplitude_r = rms_amplitude / sqrt(2.) * sqrt(log(1. / rand_uniform())) / a;

    // Left moving component
    phase = 2. * pi * rand_uniform(); // Set phase randomly
    re_f_left = amplitude_l * cos(phase);
    im_f_left = amplitude_l * sin(phase);

    // Right moving component
    phase = 2. * pi * rand_uniform(); // Set phase randomly
    re_f_right = amplitude_r * cos(phase);
    im_f_right = amplitude_r * sin(phase);

    f_k[0][0] = re_f_left + re_f_right; // Re(field)
    f_k[0][1] = im_f_left + im_f_right; // Im(field)
    pi_k[0][0] = pow(a, alpha - 3) * (-omega * (im_f_left - im_f_right) - hubble_init * f_k[0][0]); // Field derivative
    pi_k[0][1] = pow(a, alpha - 3) * (omega * (re_f_left - re_f_right) - hubble_init * f_k[0][1]);

    #endif


    if (isReal) { // For real modes set the imaginary parts to zero
        #if SIMULATE_INFLATION
            // phase = 0; // Set phase randomly
            // re_f_left = amplitude_l * cos(phase);
            // im_f_left = amplitude_l * sin(phase);

            f_k[0][0] = sqrt(pw2(f_k[0][0]) + pw2(f_k[0][1])); // Re(field)
            f_k[0][1] = 0; // Im(field)
            // pi_k[0][0] = pow(a, alpha - 3) * (-omega * im_f_left - hubble_init * f_k[0][0]); // Field derivative
            // pi_k[0][1] = pow(a, alpha - 3) * (omega * re_f_left - hubble_init * f_k[0][1]);

            pi_k[0][0] = sqrt(pw2(pi_k[0][0]) + pw2(pi_k[0][1]));
            pi_k[0][1] = 0.;
        #else
        f_k[0][1] = 0.;
        pi_k[0][1] = 0.;
        #endif
    }
}





void set_mode_from_analytic_solution(std::complex<double> fk_left, std::complex<double> pik_left, std::complex<double> fk_right, std::complex<double> pik_right, fftw_complex *f_k, fftw_complex *pi_k, bool isReal) {
    double phase;
    double f_amplitude_l, f_amplitude_r, pi_amplitude_l, pi_amplitude_r;
    double re_f_left, im_f_left, re_f_right, im_f_right;
    double re_pi_left, im_pi_left, re_pi_right, im_pi_right;

    // Left moving component
    phase = 2. * pi * rand_uniform(); // Set phase randomly
    f_amplitude_l = sqrt(log(1. / rand_uniform()));

    // phase = 0.;
    // f_amplitude_l = 1.;

    re_f_left = (fk_left.real() * cos(phase) - fk_left.imag() * sin(phase)) / sqrt(2) * f_amplitude_l;
    im_f_left = (fk_left.imag() * cos(phase) + fk_left.real() * sin(phase)) / sqrt(2) * f_amplitude_l;
    re_pi_left = (pik_left.real() * cos(phase) - pik_left.imag() * sin(phase)) / sqrt(2) * f_amplitude_l;
    im_pi_left = (pik_left.imag() * cos(phase) + pik_left.real() * sin(phase)) / sqrt(2) * f_amplitude_l;


    // Right moving component
    phase = 2. * pi * rand_uniform(); // Set phase randomly
    f_amplitude_r = sqrt(log(1. / rand_uniform()));

    // phase = 0.;
    // f_amplitude_r = 1.;

    re_f_right = (fk_right.real() * cos(phase) - fk_right.imag() * sin(phase)) / sqrt(2) * f_amplitude_r;
    im_f_right = (fk_right.imag() * cos(phase) + fk_right.real() * sin(phase)) / sqrt(2) * f_amplitude_r;
    re_pi_right = (pik_right.real() * cos(phase) - pik_right.imag() * sin(phase)) / sqrt(2) * f_amplitude_r;
    im_pi_right = (pik_right.imag() * cos(phase) + pik_right.real() * sin(phase)) / sqrt(2) * f_amplitude_r;


    // Set the field and its derivative
    // Field is set to the sum of left and right moving waves
    f_k[0][0] = (re_f_left + re_f_right) * N3 / pow(L, 1.5); // Re(field)
    f_k[0][1] = (im_f_left + im_f_right) * N3 / pow(L, 1.5); // Im(field)
    pi_k[0][0] = (re_pi_left + re_pi_right) * N3 / pow(L, 1.5);
    pi_k[0][1] = (im_pi_left + im_pi_right) * N3 / pow(L, 1.5);
    if (isReal) { // For real modes set the imaginary parts to zero
        f_k[0][1] = 0.;
        pi_k[0][1] = 0.;
    }
}


#if WITH_GW

void set_mode_GW(Vector3 k_vec, double k2, size_t k_index, fftw_complex* u_k[3][3], fftw_complex* pi_u_k[3][3], bool isReal) {
    double phase, amplitude_l, amplitude_r, rms_amplitude, k;
    double re_f_left, im_f_left, re_f_right, im_f_right;

    auto e_TT = TT_tensors(k_vec);

    static double norm = MPl * omegaStar / pw2(fStar) * pow(N / dx, 1.5);

    // Momentum cutoff. If kcutoff!=0 then eliminate all initial modes with k>kcutoff.
    static double k2cutoff = (kcutoff < 2. * pi * (double)N / L ? pw2(kcutoff) : 0.);
    if (k2cutoff > 0. && k2 > k2cutoff) {
        for (size_t i_GW = 0; i_GW < 3; i_GW++) {
            for (size_t j_GW = 0; j_GW < 3; j_GW++) {
                u_k[i_GW][j_GW][k_index][0] = 0.;
                u_k[i_GW][j_GW][k_index][1] = 0.;

                pi_u_k[i_GW][j_GW][k_index][0] = 0.;
                pi_u_k[i_GW][j_GW][k_index][1] = 0.;
            }
        }
        return;
    }

    k = sqrt(k2);
    rms_amplitude = norm / sqrt(k);

    amplitude_l = rms_amplitude / sqrt(2.) * sqrt(log(1. / rand_uniform())) / a;
    amplitude_r = rms_amplitude / sqrt(2.) * sqrt(log(1. / rand_uniform())) / a;

    // Left moving component
    phase = 2. * pi * rand_uniform(); // Set phase randomly
    re_f_left = amplitude_l * cos(phase);
    im_f_left = amplitude_l * sin(phase);
    // Right moving component
    phase = 2. * pi * rand_uniform(); // Set phase randomly
    re_f_right = amplitude_r * cos(phase);
    im_f_right = amplitude_r * sin(phase);


    for (size_t i_GW = 0; i_GW < 3; i_GW++) {
        for (size_t j_GW = 0; j_GW < 3; j_GW++) {
            u_k[i_GW][j_GW][k_index][0] = e_TT[i_GW][j_GW] * (re_f_left + re_f_right);
            u_k[i_GW][j_GW][k_index][1] = e_TT[i_GW][j_GW] * (im_f_left + im_f_right);

            pi_u_k[i_GW][j_GW][k_index][0] = pow(a, alpha - 3) * (e_TT[i_GW][j_GW] * (-k * (im_f_left - im_f_right) - hubble_init * u_k[i_GW][j_GW][k_index][0]));
            pi_u_k[i_GW][j_GW][k_index][1] = pow(a, alpha - 3) * (e_TT[i_GW][j_GW] * (k * (re_f_left - re_f_right) - hubble_init * u_k[i_GW][j_GW][k_index][1]));

            if (isReal) {
                u_k[i_GW][j_GW][k_index][1] = 0.;
                pi_u_k[i_GW][j_GW][k_index][1] = 0.;
            }
        }
    }
}


void set_conjugate_mode_GW(int conj_k_index, int k_index, fftw_complex *u_k[3][3], fftw_complex *pi_u_k[3][3]) {
    for (size_t i_GW = 0; i_GW < 3; ++i_GW) {
        for (size_t j_GW = 0; j_GW < 3; ++j_GW) {
            u_k[i_GW][j_GW][conj_k_index][0] = u_k[i_GW][j_GW][k_index][0];
            u_k[i_GW][j_GW][conj_k_index][1] = -u_k[i_GW][j_GW][k_index][1];
            pi_u_k[i_GW][j_GW][conj_k_index][0] = pi_u_k[i_GW][j_GW][k_index][0];
            pi_u_k[i_GW][j_GW][conj_k_index][1] = -pi_u_k[i_GW][j_GW][k_index][1];
        }
    }
}


void initialize_GW() {
    DECLARE_INDICES
    int fld;
    double k2; // Total squared momentum
    double dp2 = pw2(kIR); // Square of grid spacing in momentum space

    int px, py, pz; // Components of momentum in units of grid spacing

    fftw_plan backwark_plan_f;
    fftw_plan backwark_plan_fd;

    size_t k_index, conj_k_index;
    size_t i_GW, j_GW;
    Vector3 k_vec;


    for (k = 0; k <= N/2; k += N/2) { // On the k=0, N/2 plane
        for (i = 0; i <= N/2; i += N/2) {
            for (j = 1; j < N/2; j++) {
                k_vec = {i * kIR, j * kIR, k * kIR}; // (+x, +y) mode
                k2 = dp2 * (pw2(i) + pw2(j) + pw2(k));
                k_index = (i * N + j) * (N/2+1) + k;
                conj_k_index = (i * N + (N-j)) * (N/2+1) + k; // conjugate mode

                set_mode_GW(k_vec, k2, k_index, u_k, pi_u_k, false);
                set_conjugate_mode_GW(conj_k_index, k_index, u_k, pi_u_k);
            }
        }
        for (j = 0; j <= N/2; j += N/2) {
            for (i = 1; i < N/2; i++) {
                k_vec = {i * kIR, j * kIR, k * kIR}; // (+x, +y) mode
                k2 = dp2 * (pw2(i) + pw2(j) + pw2(k));
                k_index = (i * N + j) * (N/2+1) + k;
                conj_k_index = ((N-i) * N + j) * (N/2+1) + k; // conjugate mode

                set_mode_GW(k_vec, k2, k_index, u_k, pi_u_k, false);
                set_conjugate_mode_GW(conj_k_index, k_index, u_k, pi_u_k);
            }
        }
        for (i = 1; i < N/2; i++) {
            for (j = 1; j < N/2; j++) {
                k_vec = {i * kIR, j * kIR, k * kIR}; // (+x, +y) mode
                k2 = dp2 * (pw2(i) + pw2(j) + pw2(k));
                k_index = (i * N + j) * (N/2+1) + k;
                conj_k_index = ((N-i) * N + (N-j)) * (N/2+1) + k; // conjugate mode

                set_mode_GW(k_vec, k2, k_index, u_k, pi_u_k, false);
                set_conjugate_mode_GW(conj_k_index, k_index, u_k, pi_u_k);
            }
            for (j = N/2+1; j < N; j++) {
                k_vec = {i * kIR, (j - N) * kIR, k * kIR}; // (+x, -y) modenegative y mode
                k2 = dp2 * (pw2(i) + pw2(j-N) + pw2(k));
                k_index = (i * N + j) * (N/2+1) + k;
                conj_k_index = ((N-i) * N + (N-j)) * (N/2+1) + k; // conjugate mode

                set_mode_GW(k_vec, k2, k_index, u_k, pi_u_k, false);
                set_conjugate_mode_GW(conj_k_index, k_index, u_k, pi_u_k);
            }
        }
    }

    for (k = 1; k < N/2; k++) { // except the k=0, N/2 plane, no conjugate
        for (i = 0; i < N; i++) {
            px = (i <= N/2 ? i : i - N);
            for (j = 0; j < N; j++) {
                py = (j <= N/2 ? j : j - N);
                k_vec = {px * kIR, py * kIR, k * kIR};
                k2 = dp2 * (pw2(px) + pw2(py) + pw2(k));
                k_index = (i * N + j) * (N/2+1) + k;

                set_mode_GW(k_vec, k2, k_index, u_k, pi_u_k, false);
            }
        }
    }

    for (k = 0; k <= N/2; k += N/2) { // set the 8 "corners", to real
        for (i = 0; i <= N/2; i += N/2) {
            for (j = 0; j <= N/2; j += N/2) {
                if (i == 0 && j == 0 && k == 0) {
                    for (i_GW = 0; i_GW < 3; i_GW++) {
                        for (j_GW = 0; j_GW < 3; j_GW++) {
                            u_k[i_GW][j_GW][0][0] = 0;
                            u_k[i_GW][j_GW][0][1] = 0.;

                            pi_u_k[i_GW][j_GW][0][0] = 0;
                            pi_u_k[i_GW][j_GW][0][1] = 0.;
                        }
                    }
                    continue; // Skip the case where i = 0, j = 0, and k = 0
                }
                k2 = dp2 * (pw2(i) + pw2(j) + pw2(k));
                k_vec = {i * kIR, j * kIR, k * kIR};
                k_index = (i * N + j) * (N/2+1) + k;

                set_mode_GW(k_vec, k2, k_index, u_k, pi_u_k, true);
            }
        }
    }

    // *DEBUG* The option to not use the FFTs is for debugging purposes only. For actual simulations seed should always be positive
    if (seed >= 0) {
        for (i_GW = 0; i_GW < 3; i_GW++) {
            for (j_GW = 0; j_GW < 3; j_GW++) {
                std::memcpy(Cache_Data_k, u_k[i_GW][j_GW], sizeof(fftw_complex) * N * N * (N/2 + 1));
                backwark_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &u[i_GW][j_GW][0][0][0], FFTW_ESTIMATE);
                fftw_execute(backwark_plan_f);

                std::memcpy(Cache_Data_k, pi_u_k[i_GW][j_GW], sizeof(fftw_complex) * N * N * (N/2 + 1));
                backwark_plan_fd = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_u[i_GW][j_GW][0][0][0], FFTW_ESTIMATE);
                fftw_execute(backwark_plan_fd);

                for (i = 0; i < N; i++) {
                    for (j = 0; j < N; j++) {
                        for (k = 0; k < N; k++) {
                            u[i_GW][j_GW][i][j][k] /= N3;
                            pi_u[i_GW][j_GW][i][j][k] /= N3;
                        }
                    }
                }
            }
        }
    }


    fftw_destroy_plan(backwark_plan_f);
    fftw_destroy_plan(backwark_plan_fd);


    std::cout << "Finished GWs initial conditions" << std::endl;
}

#endif


void set_conjugate_mode(int conj_k_index, int k_index, fftw_complex *f_k, fftw_complex *pi_k) {
    f_k[conj_k_index][0] = f_k[k_index][0];
    f_k[conj_k_index][1] = -f_k[k_index][1];
    pi_k[conj_k_index][0]  = pi_k[k_index][0];
    pi_k[conj_k_index][1]  = -pi_k[k_index][1];
}






void checkdirectory() {

    if (!std::filesystem::exists(dir_)) {
        if (std::filesystem::create_directories(dir_)) {
            std::cout << "Directory created: \"" << dir_ << "\", data will be saved there." << std::endl;

        } else {
            std::cerr << "Failed to create directory: " << dir_ << std::endl;
            std::exit(1);
        }
    } else {
        std::cout << "Directory \"" << dir_ << "\" already exists, data will be saved there." << std::endl;
    }

    if (!std::filesystem::exists(dir_ + snap_dir_)) {
        if (std::filesystem::create_directories(dir_ + snap_dir_)) {
            std::cout << "Directory created: \"" << dir_ + snap_dir_ << "\", data will be saved there." << std::endl << std::endl;

        } else {
            std::cerr << "Failed to create directory: " << dir_ + snap_dir_ << std::endl;
            std::exit(1);
        }
    } else {
        std::cout << "Directory \"" << dir_ + snap_dir_ << "\" already exists, snapshots will be saved there." << std::endl << std::endl;
    }
}

void check_analytic_dt() {
    if(dt_analytic > 2. / (kIR * sqrt(NDIMS) * N / 2. * pow(a0, alpha - 1))) {
        std::cout << "The time step for ANALYTIC EVOLUTION is too large. The current dt_analytic is " << dt_analytic << ", but for stability in VV2 method should never exceed 2 / (kIR * sqrt(NDIMS) * N / 2. / pow(a0, alpha - 1)) = " << 2. / (kIR * sqrt(NDIMS) * N / 2. * pow(a0, alpha - 1)) << std::endl << std::endl;
        std::cout << "Adjust dt_analytic to a MAXIMUM of " << std::scientific << std::setprecision(6) << 2. / (kIR * sqrt(NDIMS) * N / 2. * pow(a0, alpha - 1)) << ", and preferably choose a slightly smaller value. And attention tOutputFreq should be evenly divisible by dt_analytic" << std::endl << std::endl;
        exit(1);
    }
    if (t_analytic > tf) {
        std::cout << "t_analytic should not larger than tf. Please reset the parameters." << std::endl;
        exit(1);
    }
}


void check_dtdx() {
    // Check to make sure time step is small enough to satisfy Courant condition, dt/dx < 1/Sqrt(ndims)
    if(dt > dx / sqrt((double)NDIMS)) {
        std::cout << "The time step for LATTICE SIMULATION is too large. The ratio dt/dx is currently " << dt / dx << " but for stability should never exceed 1/sqrt(" << NDIMS << ")=" << 1. / sqrt(static_cast<double>(NDIMS)) << std::endl << std::endl;
        std::cout << "Adjust dt to a MAXIMUM of " << std::scientific << std::setprecision(6) << dx / sqrt(static_cast<double>(NDIMS)) << ", and preferably somewhat smaller than that." << std::endl << std::endl;
        std::cout << "Or adjust kIR smaller than " << std::fixed << 2 * M_PI / sqrt(static_cast<double>(NDIMS)) / N / dt << ", and preferably somewhat smaller than that." << std::endl << std::endl;
        exit(1);
    }
}




// double ad_only_backgroud(const double a, double (&f)[nflds], const double (&pi_f)[nflds]) {
//     double ad;
//     double K_E = 0;
//     double G_E = 0;
//     double V_E = 0;

//     for (size_t fld = 0; fld < nflds; fld++) {
//         K_E += pw2(pi_f[fld]);

//         // G_E += gradient_energy(a, f[fld]);
//     }
//     K_E /=  2. * pow(a, 6);

//     for (size_t term = 0; term < num_potential_terms; term++) {
//         V_E += potential_energy(term, f);
//     }

//     ad = pow(a, 1 + alpha) * (fStar / MPl) * sqrt((K_E + G_E + V_E) / 3.);
//     return ad;
// }


void initialize() {
    DECLARE_INDICES
    int fld;
    double p2; // Total squared momentum
    double dp2 = pw2(kIR); // Square of grid spacing in momentum space
    double mass_sq[nflds]; // Effective mass squared of fields
    double initial_f[nflds]; // Initial value of fields (set to zero unless specified in parameters.h)
    double initial_fd[nflds]; // Initial value of field derivatives (set to zero unless specified in parameters.h)
    double initial_pi[nflds];
    double fdsquared = 0., pot_energy = 0.; // Sum(field_dot^2), and potential energy - used for calculating the Hubble constant
    std::ifstream old_grid_("output/grid.img", std::ios::binary);

    int px, py, pz; // Components of momentum in units of grid spacing

    #if !ANALYTIC_EVOLUTION
    // check_dtdx();
    #endif

    fftw_plan backwark_plan_f;
    fftw_plan backwark_plan_fd;

    modelinitialize(1); // This allows specific models to perform any needed initialization

    // Output initializations - Set values of nfldsout and ext_
    // if(alt_extension[0]!='\0') // If an alternate extension was given use that instead of the default "_<run_number>.dat"
    //     ext_ = alt_extension;
        // snprintf(ext_,sizeof(ext_), "%s", alt_extension);

    if (continue_run && old_grid_.is_open()) {
        std::cout << "Previously generated grid image found. Reading in data..." << std::endl;

        old_grid_.read(reinterpret_cast<char*>(&run_number), sizeof(run_number));
        run_number++;
        old_grid_.read(reinterpret_cast<char*>(&t0), sizeof(t0));

        if (t0 >= tf) {
            std::cout << "A grid image file was found in this directory with values stored at t=" << t0
                      << ". To continue that run set tf to a later time. To start a new run move or rename the file grid.img." << std::endl;
            std::exit(1);
        }

        old_grid_.read(reinterpret_cast<char*>(&a), sizeof(a));
        old_grid_.read(reinterpret_cast<char*>(&ad), sizeof(ad));
        old_grid_.read(reinterpret_cast<char*>(f), sizeof(double) * nflds * N3);
        old_grid_.read(reinterpret_cast<char*>(pi_f), sizeof(double) * nflds * N3);

        old_grid_.close();

        // if (alt_extension.empty()) {
        //     ext_ = "_" + std::to_string(run_number) + ".dat";
        // }

        std::cout << "Data read. Resuming run at t = " << t0 << std::endl;
        output_parameters(); // Save information about the model and parameters and start the clock for timing the run
    }


    // If no old grid image is found generate new initial conditions and set run_number=0
    std::cout << "Generating lattice initial conditions for new run at t = 0" << std::endl;
    t0 = 0;
    run_number = 0;

    for (fld = 0; fld < nflds; fld++) {
        // Set initial field values
        if (fld < (int)(sizeof initfield / sizeof(double))) // Use preset values for initial field values if given
            initial_f[fld] = initfield[fld] / fStar;
        else // Otherwise initialize field to zero
            initial_f[fld] = 0.;
        
        // Set initial field derivative values
        if (fld < (int)(sizeof initderivs / sizeof(double))) { // Use preset values for initial field derivative values if given
            initial_fd[fld] = initderivs[fld] / fStar / omegaStar;
            initial_pi[fld] = pow(a, 3) * initderivs[fld] / fStar / omegaStar;
        }
        else { // Otherwise initialize derivatives to zero
            initial_fd[fld] = 0.;
            initial_pi[fld] = 0.;
        }
    }


    // Set initial values of effective mass.
    effective_mass(mass_sq, initial_f);
    // mass_sq[0] = d2Vdf2(0, initial_f);
    // mass_sq[1] = d2Vdf2(2, initial_f);

    // Set initial value of Hubble constant - See the documentation for an explanation of the formulas used
    if (expansion > 0) {
        if (expansion == 1)
            std::cout << "The initial value of the fields is being used to determine the initial Hubble constant.\nFrom here on power law expansion will be used" << std::endl;
        for (fld = 0; fld < nflds; fld++) // Find sum of squares of derivatives
            fdsquared += pw2(initial_fd[fld]);
        for (i = 0; i < num_potential_terms; i++) // Find potential energy
            pot_energy += potential_energy(i, initial_f);
        std::cout << "The initial kinetic energy (sum of squares of field derivatives / 2) is: " << fdsquared / 2. << "\nThe initial potential energy is: " << pot_energy << std::endl;
        hubble_init = fdsquared / 2.; // Kinetic energy
        hubble_init += pot_energy; // Potential energy
        hubble_init *= pw2(fStar / MPl) / 3.;
        hubble_init = sqrt(hubble_init); // with respect to program time \tilde{eta}
        if (!(hubble_init >= 0.)) { // Make sure Hubble isn't negative or undefined
            std::cout << "Error in calculating initial Hubble constant. Exiting." << std::endl;
            exit(1);
        }
        a = a0;
        ad = hubble_init * a;
        add = pw2(fStar / MPl) / 3. * ((alpha - 2) * fdsquared + (alpha + 1) * pot_energy) * a;
    }

    std::cout << "initial a = " << a << ", ad = " << ad << ", add = " << add << std::endl;


    if (pw2(kIR) < 10 * add / pow(a, 3) || kIR < hubble_init) {
        std::cout << "If you are simulating inflation, the smallest length mode kIR should be larger than sqrt(10 * a'' / a) = " << sqrt(10 * add / a) << ", and kIR should larger than aH = " << ad << " . However, kIR = " << kIR << " do not meet this criterion, indicating that it is not a Bunch-Davies vacuum. Please set kIR larger than " << std::max(sqrt(10 * add / a), ad) << std::endl << std::endl;

    } else {
        std::cout << "If you are simulating inflation, No k ^ 2 values are smaller than 10 a'' / a." << std::endl << std::endl;
    }

    std::cout << "Press ENTER to continue";
    std::cin.get(); // Pause and wait for Enter


    size_t k_index, conj_k_index;
    for (fld = 0; fld < nflds; fld++) {
        for (k = 0; k <= N/2; k += N/2) { // On the k=0, N/2 plane
            for (i = 0; i <= N/2; i += N/2) {
                for (j = 1; j < N/2; j++) {
                    p2 = dp2 * (pw2(i) + pw2(j) + pw2(k));
                    k_index = (i * N + j) * (N/2+1) + k;
                    conj_k_index = (i * N + (N-j)) * (N/2+1) + k;

                    set_mode(p2, mass_sq[fld], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                    set_conjugate_mode(conj_k_index, k_index, f_k[fld], pi_k[fld]);
                }
            }
            for (j = 0; j <= N/2; j += N/2) {
                for (i = 1; i < N/2; i++) {
                    p2 = dp2 * (pw2(i) + pw2(j) + pw2(k));
                    k_index = (i * N + j) * (N/2+1) + k;
                    conj_k_index = ((N-i) * N + j) * (N/2+1) + k;

                    set_mode(p2, mass_sq[fld], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                    set_conjugate_mode(conj_k_index, k_index, f_k[fld], pi_k[fld]);
                }
            }
            for (i = 1; i < N/2; i++) {
                for (j = 1; j < N/2; j++) {
                    p2 = dp2 * (pw2(i) + pw2(j) + pw2(k));
                    k_index = (i * N + j) * (N/2+1) + k;
                    conj_k_index = ((N-i) * N + (N-j)) * (N/2+1) + k;

                    set_mode(p2, mass_sq[fld], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                    set_conjugate_mode(conj_k_index, k_index, f_k[fld], pi_k[fld]);
                }
                for (j = N/2+1; j < N; j++) {
                    p2 = dp2 * (pw2(i) + pw2(j-N) + pw2(k));
                    k_index = (i * N + j) * (N/2+1) + k;
                    conj_k_index = ((N-i) * N + (N-j)) * (N/2+1) + k;

                    set_mode(p2, mass_sq[fld], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                    set_conjugate_mode(conj_k_index, k_index, f_k[fld], pi_k[fld]);
                }
            }
        }

        for (k = 1; k < N/2; k++) { // except the k=0, N/2 plane, no conjugate
            for (i = 0; i < N; i++) {
                px = (i <= N/2 ? i : i - N);
                for (j = 0; j < N; j++) {
                    py = (j <= N/2 ? j : j - N);
                    p2 = dp2 * (pw2(px) + pw2(py) + pw2(k));
                    k_index = (i * N + j) * (N/2+1) + k;

                    set_mode(p2, mass_sq[fld], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                }
            }
        }

        for (k = 0; k <= N/2; k += N/2) { // set the 8 "corners", to real
            for (i = 0; i <= N/2; i += N/2) {
                for (j = 0; j <= N/2; j += N/2) {
                    p2 = dp2 * (pw2(i) + pw2(j) + pw2(k));
                    k_index = (i * N + j) * (N/2+1) + k;

                    set_mode(p2, mass_sq[fld], &f_k[fld][k_index], &pi_k[fld][k_index], true);
                }
            }
        }


        f_k[fld][0][0] = initial_f[fld] * N3;
        f_k[fld][0][1] = 0.;

        pi_k[fld][0][0] = pow(a, alpha - 3) * initial_fd[fld] * N3;
        pi_k[fld][0][1] = 0.;

        if (seed >= 0) {
            std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            backwark_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(backwark_plan_f);

            std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
            backwark_plan_fd = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
            fftw_execute(backwark_plan_fd);

            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    for (k = 0; k < N; k++) {
                        f[fld][i][j][k] /= N3;
                        pi_f[fld][i][j][k] /= N3;
                    }
                }
            }
        }
    } // End loop over fields



    #if !SPECTRAL_FLAG
        // add = kernel_a(1, f, pi_f);
        add = kernel_a();
    #else
        #if PAD_FLAG
            for (size_t fld = 0; fld < nflds; fld++) {
                memset(f_k_pad[fld], 0, sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
                padding(f_k[fld], f_k_pad[fld]);
                std::memcpy(Cache_Data_k_pad, f_k_pad[fld], sizeof(fftw_complex) * N_pad * N_pad * (N_pad/2 + 1));
                plan = fftw_plan_dft_c2r_3d(N_pad, N_pad, N_pad, Cache_Data_k_pad, &f_pad[fld][0][0][0], FFTW_ESTIMATE);
                fftw_execute(plan);

                LOOP_pad
                    f_pad[fld][i][j][k] /= N3;
            }
            add = kernel_a_k_pad(1, f, f_pad, pi_k);
        #else
            add = kernel_a_k_nopad();
            // add = kernel_a_only_backgroud(a, initial_f, initial_fd);
        #endif
    #endif

    fftw_destroy_plan(backwark_plan_f);
    fftw_destroy_plan(backwark_plan_fd);

    modelinitialize(2); // This allows specific models to perform any needed initialization
    // save(); // Save field values and derived quantities
    output_parameters(); // Save information about the model and parameters and start the clock for timing the run

    std::cout << "Finished initial conditions" << std::endl;
}



#if ANALYTIC_EVOLUTION
void initialize_analytic_perturbation() {
    int k;
    int fld;
    // double k_mode;
    double norm = omegaStar / fStar;
    double mass_sq[nflds];
    double omega[nflds], real[nflds], imag[nflds];
    double initial_f[nflds];
    double initial_fd[nflds];
    double initial_pi[nflds];


    ///////////////////////////////////////////////////////////////////////////
    // check_dtdx();
    check_analytic_dt();
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////


    // If no old grid image is found generate new initial conditions and set run_number=0
    std::cout << "Generating analytic initial conditions for new run at t = " << t << std::endl;
    t = t0;

    a_k = a0;

    for (fld = 0; fld < nflds; fld++) {
        // Set initial field values
        if (fld < (int)(sizeof initfield / sizeof(double))) // Use preset values for initial field values if given
            initial_f[fld] = initfield[fld] / fStar;
        else // Otherwise initialize field to zero
            initial_f[fld] = 0.;

        // Set initial field derivative values
        if (fld < (int)(sizeof initderivs / sizeof(double))) { // Use preset values for initial field derivative values if given
            initial_fd[fld] = initderivs[fld] / fStar / omegaStar;
            initial_pi[fld] = pow(a_k, 3) * initderivs[fld] / fStar / omegaStar;
        }
        else { // Otherwise initialize derivatives to zero
            initial_fd[fld] = 0.;
            initial_pi[fld] = 0.;
            std::cout << "pi[" << fld << "] is 0" << std::endl;
        }

        std::cout << "initial_pi["<< fld << "]=" << initial_pi[fld] << std::endl;
    }


    double fdsquared = 0.; // Sum of squares of derivatives
    double pot_energy = 0.; // Potential energy

    if (expansion > 0) {
        if (expansion == 1)
            std::cout << "The initial value of the fields is being used to determine the initial Hubble constant.\nFrom here on power law expansion will be used" << std::endl;
        for (fld = 0; fld < nflds; fld++) // Find sum of squares of derivatives
            fdsquared += pw2(initial_pi[fld]);
        for (size_t term = 0; term < num_potential_terms; term++) // Find potential energy
            pot_energy += potential_energy(term, initial_f);
        fdsquared /= 2. * pow(a_k, 6);
        hubble_init = fdsquared; // Kinetic energy
        hubble_init += pot_energy; // Potential energy
        hubble_init *= pow(a_k, 2 * alpha) / 3. * pw2(fStar / MPl);
        std::cout << "HI ^ 2 = " << hubble_init << std::endl;
        hubble_init = sqrt(hubble_init); // with respect to program time \tilde{eta}
        if (!(hubble_init >= 0.)) { // Make sure Hubble isn't negative or undefined
            std::cout << "Error in calculating initial Hubble constant. Exiting." << std::endl;
            exit(1);
        }
        ad_k = hubble_init * a_k;
        add_k = kernel_a_only_backgroud(a_k, initial_f, initial_pi);
    }
    std::cout << "fdsquared = " << fdsquared << std::endl;
    std::cout << "pot_energy = " << pot_energy << std::endl;
    std::cout << "HI = " << hubble_init << std::endl;

    // Set initial values of effective mass.
    effective_mass(mass_sq, initial_f);
    
    std::cout << "initial a = " << a_k << ", ad = " << ad_k << ", add = " << add_k << std::endl;


    if (pw2(kIR) < 10 * add_k / pow(a_k, 3) || kIR < hubble_init) {
        std::cout << "If you are simulating inflation, the smallest length mode kIR should be larger than sqrt(10 * a'' / a) = " << sqrt(10 * add_k / a_k) << ", and kIR should larger than aH = " << ad_k << " . However, kIR = " << kIR << " do not meet this criterion, indicating that it is not a Bunch-Davies vacuum. Please set kIR larger than " << std::max(sqrt(10 * add_k / a_k), ad_k) << std::endl << std::endl;

        std::cout << "If you still want to proceed, press ENTER";
        std::cin.get(); // Pause and wait for Enter
    } else {
        std::cout << "If you are simulating inflation, No k ^ 2 values are smaller than 10 a'' / a." << std::endl << std::endl;
    }



    fk_left.resize(nflds);
    pi_k_left.resize(nflds);

    fk_right.resize(nflds);
    pi_k_right.resize(nflds);

    for (fld = 0; fld < nflds; fld++) {
        fk_left[fld].resize(num_k_values);
        pi_k_left[fld].resize(num_k_values);

        fk_right[fld].resize(num_k_values);
        pi_k_right[fld].resize(num_k_values);
    }

    for (k = 1; k < num_k_values; k++) {
        double p2 = k_square_vec[k] * pw2(kIR);
        // double p2 = k_square_vec[k] * pw2(kOverA);

        for (fld = 0; fld < nflds; fld++) {
            omega[fld] = sqrt(p2 + pw2(a_k) * mass_sq[fld]);

            real[fld] = cos(omega[fld] * t0) / sqrt(2 * omega[fld]) / a_k;
            imag[fld] = sin(omega[fld] * t0) / sqrt(2 * omega[fld]) / a_k;

            fk_left[fld][k] = norm * std::complex<double>(real[fld], imag[fld]);
            pi_k_left[fld][k] = pow(a_k, 3 - alpha) * norm * std::complex<double>(- imag[fld] * omega[fld] * pow(a_k, alpha - 1) - hubble_init * real[fld], real[fld] * omega[fld] * pow(a_k, alpha - 1) - hubble_init * imag[fld]);

            fk_right[fld][k] = norm * std::complex<double>(real[fld], imag[fld]);
            pi_k_right[fld][k] = pow(a_k, 3 - alpha) * norm * std::complex<double>(imag[fld] * omega[fld] * pow(a_k, alpha - 1) - hubble_init * real[fld], - real[fld] * omega[fld] * pow(a_k, alpha - 1) - hubble_init * imag[fld]);
        }
    }

    // std::cout << "k2=" << k_square_vec[1] * pw2(kIR) << std::endl;;
    // for (fld = 0; fld < nflds; fld++) {
    //     std::cout << "mass["<< fld << "]=" << mass_sq[fld] << std::endl;
    //     std::cout << "omega["<< fld << "]=" << sqrt(k_square_vec[1] * pw2(kIR) + pw2(a_k) * mass_sq[fld]) << std::endl;
    //     std::cout << "f_k[" << fld << "][k=25]=" << fk_left[fld][1] << ", |f_k| = " << sqrt(pw2(fk_left[fld][1].real()) + pw2(fk_left[fld][1].imag())) << ", \\Delta = " << pow(kIR, 3) / (2 * M_PI) * pw2(fk_left[fld][1].real()) + pw2(fk_left[fld][1].imag()) << std::endl;
    //     std::cout << "pi_k[" << fld << "][k=25]=" << pi_k_left[fld][1] << std::endl;
    // }
    // std::cin.get();

    // background
    for (fld = 0; fld < nflds; fld++) {
        fk_left[fld][0] = std::complex<double>(initial_f[fld], 0.);
        pi_k_left[fld][0] = std::complex<double>(initial_pi[fld], 0.);

        fk_right[fld][0] = std::complex<double>(initial_f[fld], 0.);
        pi_k_right[fld][0] = std::complex<double>(initial_pi[fld], 0.);
    }

    save_analytic();
}



void initialize_lattice() {
    DECLARE_INDICES
    size_t fld;
    double p2; // Total squared momentum
    int k2;
    double dp2 = pw2(kIR); // Square of grid spacing in momentum space
    double px, py; // Components of momentum in units of grid spacing

    std::size_t position;

    fftw_plan backwark_plan_f;
    fftw_plan backwark_plan_fd;

    a = a_k;
    ad = ad_k;
    add = add_k;

    std::cout << std::endl << "Generating lattice initial conditions for new run at t = " << t << std::endl;

    int k_index, conj_k_index;
    for (fld = 0; fld < nflds; fld++) {
        for (k = 0; k <= N/2; k += N/2) { // On the k=0, N/2 plane
            for (i = 0; i <= N/2; i += N/2) {
                for (j = 1; j < N/2; j++) {
                    k2 = pw2(i) + pw2(j) + pw2(k);

                    auto it = k_square.find(k2);
                    position = std::distance(k_square.begin(), it);

                    k_index = (i * N + j) * (N/2+1) + k;
                    conj_k_index = (i * N + (N-j)) * (N/2+1) + k;

                    set_mode_from_analytic_solution(fk_left[fld][position], pi_k_left[fld][position], fk_right[fld][position], pi_k_right[fld][position], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                    set_conjugate_mode(conj_k_index, k_index, f_k[fld], pi_k[fld]);
                }
            }
            for (j = 0; j <= N/2; j += N/2) {
                for (i = 1; i < N/2; i++) {
                    k2 = pw2(i) + pw2(j) + pw2(k);

                    auto it = k_square.find(k2);
                    position = std::distance(k_square.begin(), it);

                    k_index = (i * N + j) * (N/2+1) + k;
                    conj_k_index = ((N-i) * N + j) * (N/2+1) + k;

                    set_mode_from_analytic_solution(fk_left[fld][position], pi_k_left[fld][position], fk_right[fld][position], pi_k_right[fld][position], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                    set_conjugate_mode(conj_k_index, k_index, f_k[fld], pi_k[fld]);
                }
            }
            for (i = 1; i < N/2; i++) {
                for (j = 1; j < N/2; j++) {
                    k2 = pw2(i) + pw2(j) + pw2(k);

                    auto it = k_square.find(k2);
                    position = std::distance(k_square.begin(), it);

                    k_index = (i * N + j) * (N/2+1) + k;
                    conj_k_index = ((N-i) * N + (N-j)) * (N/2+1) + k;

                    set_mode_from_analytic_solution(fk_left[fld][position], pi_k_left[fld][position], fk_right[fld][position], pi_k_right[fld][position], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                    set_conjugate_mode(conj_k_index, k_index, f_k[fld], pi_k[fld]);
                }
                for (j = N/2+1; j < N; j++) {
                    k2 = pw2(i) + pw2(j-N) + pw2(k);

                    auto it = k_square.find(k2);
                    position = std::distance(k_square.begin(), it);

                    k_index = (i * N + j) * (N/2+1) + k;
                    conj_k_index = ((N-i) * N + (N-j)) * (N/2+1) + k;

                    set_mode_from_analytic_solution(fk_left[fld][position], pi_k_left[fld][position], fk_right[fld][position], pi_k_right[fld][position], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                    set_conjugate_mode(conj_k_index, k_index, f_k[fld], pi_k[fld]);
                }
            }
        }

        for (k = 1; k < N/2; k++) { // except the k=0, N/2 plane, no conjugate
            for (i = 0; i < N; i++) {
                px = (i <= N/2 ? i : i - N);
                for (j = 0; j < N; j++) {
                    py = (j <= N/2 ? j : j - N);
                    k2 = pw2(px) + pw2(py) + pw2(k);

                    auto it = k_square.find(k2);
                    position = std::distance(k_square.begin(), it);

                    k_index = (i * N + j) * (N/2+1) + k;

                    set_mode_from_analytic_solution(fk_left[fld][position], pi_k_left[fld][position], fk_right[fld][position], pi_k_right[fld][position], &f_k[fld][k_index], &pi_k[fld][k_index], false);
                }
            }
        }

        for (k = 0; k <= N/2; k += N/2) { // set the 8 "corners", to real
            for (i = 0; i <= N/2; i += N/2) {
                for (j = 0; j <= N/2; j += N/2) {
                    k2 = pw2(i) + pw2(j) + pw2(k);

                    auto it = k_square.find(k2);
                    position = std::distance(k_square.begin(), it);

                    k_index = (i * N + j) * (N/2+1) + k;

                    set_mode_from_analytic_solution(fk_left[fld][position], pi_k_left[fld][position], fk_right[fld][position], pi_k_right[fld][position], &f_k[fld][k_index], &pi_k[fld][k_index], true);
                }
            }
        }


        f_k[fld][0][0] = fk_left[fld][0].real() * N3;
        f_k[fld][0][1] = 0.;

        pi_k[fld][0][0] = pi_k_left[fld][0].real() * N3;
        pi_k[fld][0][1] = 0.;

        std::memcpy(Cache_Data_k, f_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
        backwark_plan_f = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &f[fld][0][0][0], FFTW_ESTIMATE);
        fftw_execute(backwark_plan_f);

        std::memcpy(Cache_Data_k, pi_k[fld], sizeof(fftw_complex) * N * N * (N/2 + 1));
        backwark_plan_fd = fftw_plan_dft_c2r_3d(N, N, N, Cache_Data_k, &pi_f[fld][0][0][0], FFTW_ESTIMATE);
        fftw_execute(backwark_plan_fd);


        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    f[fld][i][j][k] /= N3;
                    pi_f[fld][i][j][k] /= N3;
                }
            }
        }
    } // End loop over fields

    fftw_destroy_plan(backwark_plan_f);
    fftw_destroy_plan(backwark_plan_fd);

    #if BIFURCATION
    for (k = 1; k < maxnumbins; k++) {
        auto it = k_square.find(pw2(k));
        size_t position = std::distance(k_square.begin(), it);

        for (fld = 0; fld < nflds; fld++) {
            fk_positive[fld][k] = fk_left[fld][position];
            pi_k_positive[fld][k] = pi_k_left[fld][position];

            fk_negative[fld][k] = fk_left[fld][position];
            pi_k_negative[fld][k] = pi_k_left[fld][position];
        }
    }
    #endif

    std::cout << "Initial conditions for the lattice have been configured." << std::endl;
}



void k_square_initialize(){
    for (size_t i = 0; i < N/2+1; ++i) {
        for (size_t j = 0; j < N/2+1; ++j) {
            for (size_t k = 0; k < N/2+1; ++k) {
                int value = pw2(i) + pw2(j) + pw2(k);
                k_square.insert(value);
            }
        }
    }
    for (size_t i = 0; i < maxnumbins; ++i)
        k_square.insert(pw2(i));

    k_square_vec.assign(k_square.begin(), k_square.end());
    num_k_values = k_square_vec.size();

    // std::cout << "largest k = " << k_square_vec.back() << std::endl;
    // std::cin.get();
}

#endif



#if BIFURCATION
void initialize_k(void) {
    int k;
    int fld;
    double k_mode;
    double norm = omegaStar / fStar;
    double mass_sq[nflds];
    double omega[nflds], real[nflds], imag[nflds];
    double initial_f[nflds];
    double initial_fd[nflds];
    double initial_pi[nflds];

    for (fld = 0; fld < nflds; fld++) {
        // Set initial field values
        if (fld < (int)(sizeof initfield / sizeof(double))) // Use preset values for initial field values if given
            initial_f[fld] = initfield[fld] / fStar;
        else // Otherwise initialize field to zero
            initial_f[fld] = 0.;
        
        // Set initial field derivative values
        if (fld < (int)(sizeof initderivs / sizeof(double))) { // Use preset values for initial field derivative values if given
            initial_fd[fld] = initderivs[fld] / fStar / omegaStar;
            initial_pi[fld] = pow(a, 3) * initderivs[fld] / fStar / omegaStar;
        }
        else { // Otherwise initialize derivatives to zero
            initial_fd[fld] = 0.;
            initial_pi[fld] = 0.;
        }
    }

    // Set initial values of effective mass.
    effective_mass(mass_sq, initial_f);

    for (k = 1; k < maxnumbins; k++) {
        k_mode = k * kIR;

        for (fld = 0; fld < nflds; fld++) {
            omega[fld] = sqrt(pw2(k_mode) + pw2(a) * mass_sq[fld]);

            real[fld] = cos(omega[fld] * t0) / sqrt(2 * omega[fld]) / a;
            imag[fld] = sin(omega[fld] * t0) / sqrt(2 * omega[fld]) / a;

            fk_positive[fld][k] = norm * std::complex<double>(real[fld], imag[fld]);
            pi_k_positive[fld][k] = pow(a, 3 - alpha) * norm * std::complex<double>(- imag[fld] * omega[fld] * pow(a, alpha - 1) - hubble_init * real[fld], real[fld] * omega[fld] * pow(a, alpha - 1) - hubble_init * imag[fld]);

            fk_negative[fld][k] = norm * std::complex<double>(real[fld], imag[fld]);
            pi_k_negative[fld][k] = pow(a, 3 - alpha) * norm * std::complex<double>(- imag[fld] * omega[fld] * pow(a, alpha - 1) - hubble_init * real[fld], real[fld] * omega[fld] * pow(a, alpha - 1) - hubble_init * imag[fld]);
        }
    }

}
#endif