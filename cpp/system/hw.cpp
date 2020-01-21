#include <cstdlib>
#include <string>

using namespace std;

int main() {
	string cmd = "touch $(pwd)/../test.txt";

	system(cmd.c_str());

	return 0;
}
