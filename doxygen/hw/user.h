#ifndef USER_H
#define USER_H

/// @brief This class is used to demonstrate the usage 
/// of Doxygen
///
/// Comments starting from a new paragraph are considered detailed description
class User {
public:

    /*! \brief Initializes a user with the given id
     *                                                      */
    User(int id,               ///< [in] identification number of the user
         bool is_active = true ///< [in] status of the user
        ) : id_(id), is_active_(is_active) {}

    /// Gets the identification number of the user
    int id() const { return id_; }                   
    int is_active() const { return is_active_; }                   

private:
    ///@{ @name user information
    int id_;        ///< identification number of the user
    int is_active_; ///< active status of the user
    ///@}
};

    /*! \brief Initializes a user with the given id
     *
     * A demonstration of the usage of latex within Doxygen.
     *
     * To embed some equation in text: \f$\langle\mu|\hat{O}|p\rangle=O_{\mu\nu}\f$
     *
     * To have a one-liner:
     * \f[
     *      g(k) = \int_0^{\infty} \mathrm{d}r f(r) j_l(kr)r^2
     * \f]
     *
     * To have a multi-line equation, one needs to explicit
     * invoke some math enviroment:
     * \f{eqnarray*}{
     *      g(k) &= \int_0^{\infty} \mathrm{d}r f(r) j_l(kr)r^2\\
     *           &= \int_0^{\infty} \mathrm{d}r f(r) j_l(kr)r^2
     * \f}
     *
     *                                                      */
#endif
