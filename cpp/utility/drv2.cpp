#include "grad.h"
#include "drv2.h"

std::function<double(double)> drv2(std::function<double(double)> f ) {
	return grad(grad(f));
}


