// Copyright (c) 2024 Jie Jiang (江捷) <jiejiang@pusan.ac.kr>
// This code is licensed under the MIT License.
// See the LICENSE file in the project root for license information.


#include<cmath>
// Macros to make the equations more readable: The values of fld are 0=Phi
#define PHI FIELD(0)

inline void modelinfo_(std::ofstream &info_) {
    // Name and description of model
    info_ << "single phi^2 Model\n";
    info_ << "V = 1/2 * m_in_MPl^2 * PHI^2 \n\n";

    info_ << "m_in_MPl = " << m_in_MPl << "\n";

    info_ << "fStar = " << fStar << "\n";
    info_ << "omegaStar = " << omegaStar << "\n";
}

// Perform any model specific initialization
// This function is called twice, once before initializing the fields (which_call=1) and once after (which_call=2)
inline void modelinitialize(int which_call) {
    if (which_call == 1) {
        if (nflds != 1) {
            // printf("Number of fields for TWOFLDLAMBDA model must be 2. Exiting.\n");
            std::cout << "Number of fields for TWOFLDLAMBDA model must be 2. Exiting." << std::endl;
            exit(1);
        }
    }
}


const int num_potential_terms = 1;
inline double potential_energy(int term, double *field_values) {
    DECLARE_INDICES
    double potential = 0.;

    // if (field_values == NULL) { // If no values are given calculate averages on the lattice
    if (field_values == NULL) { // If no values are given calculate averages on the lattice
        // Loop over grid to calculate potential term
        LOOP {
            switch (term) {
                case 0:
                    potential += pw2(PHI);
                    break;
                default:
                    throw std::invalid_argument("Invalid term value in potential_energy");
            }
        }
    // Convert sum to average
    potential /= N3;
    }
    else { // If field values are given then use them instead
        switch (term) {
            case 0:
                potential = pw2(field_values[0]);
                break;
            default:
                throw std::invalid_argument("Invalid term value in potential_energy");
        }
    }

    switch (term) {
        case 0:
            potential *= 0.5;
            break;
        default:
            throw std::invalid_argument("Invalid term value in potential_energy");
    }

    return(potential);
}

inline double dvdf(int fld, const double *field_values) {
    switch (fld) {
        case 0:
            return field_values[0];
        default:
            throw std::invalid_argument("Invalid term value in dvdf");
    }
}



inline double d2Vdf2(size_t term, double *field_values) {
    switch (term) {
        case 0: // d2V / dphi dphi
            return 1;
        default:
            throw std::invalid_argument("Invalid term value in d2Vdf2 ");
    }
}


inline void effective_mass(double mass_sq[], double *field_values) {
    mass_sq[0] = 1;
}


// Model-specific output functions
inline void model_output(std::string ext_){}

#undef PHI
