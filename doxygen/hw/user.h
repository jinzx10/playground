#ifndef USER_H
#define USER_H

/// @brief This class is used to demonstrate the usage of Doxygen
/// If you start a new line, Detailed description starts from here.
class User {
public:

    User(int id ///< user's identification number
        ): id_(id) {}
    //!< Initializes a user with the given name, id and active status

    /*! get the user's identification number
     *                                                      */
    int id() const { return id_; }                   

private:
    int id_; ///< identification number of the user
};

#endif
///** Detailed explanation starts here.
// *                                                          */
