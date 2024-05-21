// Julia headers (for initialization and gc commands)
#include "uv.h"
#include "julia.h"


// prototype of the C entry points in our application
int madnlp_c(double x0, double y0, double z0, size_t iters);
