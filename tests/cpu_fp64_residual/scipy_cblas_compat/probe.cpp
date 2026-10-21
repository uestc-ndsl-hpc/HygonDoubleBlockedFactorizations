#include <cblas.h>
#include <iostream>
extern "C" const char* scipy_openblas_get_config();
extern "C" int scipy_openblas_get_num_threads();
int main() {
    const double a = 1.0 + 0x1p-23;
    double c = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 1.0, &a, 1, &a, 1, 0.0, &c, 1);
    std::cout << scipy_openblas_get_config() << '\n'
              << "threads=" << scipy_openblas_get_num_threads() << '\n'
              << std::hexfloat << "product=" << c << '\n';
    return c == a * a && c > 1.0 + 0x1p-22 ? 0 : 1;
}
