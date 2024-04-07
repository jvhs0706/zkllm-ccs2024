#ifndef POLYNOMIAL_CUH
#define POLYNOMIAL_CUH

#include "fr-tensor.cuh"

// This part is for light-weight "CPU" serial computations, which actually piggy-backs on the GPU implementations

Fr_t operator+(const Fr_t& a, const Fr_t& b);
Fr_t operator-(const Fr_t& a, const Fr_t& b);
Fr_t operator-(const Fr_t& a);
Fr_t operator*(const Fr_t& a, const Fr_t& b);
Fr_t operator/(const Fr_t& a, const Fr_t& b);
Fr_t inv(const Fr_t& a);

class Polynomial{
public:
    Polynomial();
    Polynomial(int degree);
    Polynomial(int degree, Fr_t* coefficients);
    Polynomial(const Polynomial& other);
    Polynomial(const Fr_t& constant);
    Polynomial(const vector<Fr_t>& coefficients);
    ~Polynomial();

    Polynomial& operator=(const Polynomial& other);
    Polynomial operator+(const Polynomial& other);
    Polynomial operator-(const Polynomial& other);
    Polynomial operator*(const Polynomial& other);
    Polynomial operator-();

    Polynomial& operator+=(const Polynomial& other);
    Polynomial& operator-=(const Polynomial& other);
    Polynomial& operator*=(const Polynomial& other);

    Fr_t operator()(const Fr_t& x);

    int getDegree() const;
    void setCoefficients(int degree, Fr_t* coefficients);

    static Polynomial eq(const Fr_t& u);
    static Fr_t eq(const Fr_t& u, const Fr_t& v);

    friend std::ostream& operator<<(std::ostream& os, const Polynomial& poly);
private:
    Fr_t* coefficients_;
    int degree_;
};

// Polynomial eq(const Fr_t& u);
// Fr_t eq(const Fr_t& u, const Fr_t& v);

#endif