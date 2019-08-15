#include <fib.h>
#include <fact.h>
#include <dfact.h>
#include <tq.h>
#include <iostream>

using namespace std;

int main() {
    cout << boolalpha << "fib(8) = " << fib(8) << endl
	 << "fact(5) = " << fact(5) << endl
	 << "dfact(6) = " << dfact(6) << endl
	 << "tq(false) = " << tq(false) << endl;
    
    return 0;
}

