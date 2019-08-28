from cython cimport floating


ctypedef fused integral_1:
    signed char
    short
    int
    long
    long long


ctypedef fused integral_2:
    signed char
    short
    int
    long
    long long


ctypedef fused integral_3:
    signed char
    short
    int
    long
    long long


ctypedef fused numeric:
    signed char
    short
    int
    long
    long long
    float
    double
