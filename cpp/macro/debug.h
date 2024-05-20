#ifndef __DEBUG_H__
#define __DEBUG_H__


#ifdef NDEBUG
#define PRINT_LINE_AND_FILE() (void)0
#else
#include <iostream>
#define PRINT_LINE_AND_FILE() std::cout << __LINE__ << " " << __FILE__ << std::endl;
#endif

#endif
