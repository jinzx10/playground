#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>

using namespace std;

vector<int> warmtemp(vector<int>& T) {
	int high = T.back();
	size_t sz = T.size();
	vector<int> days(sz, 0);
	days[sz-1] = 0;
	for (int i = sz-2; i >= 0; --i) {
		int Ti = T[i];
		if (Ti >= high) {
			days[i] = 0;
			high = Ti;
		} else {
			int idx = i+1;
			while (T[idx] <= Ti)
				idx += days[idx];
			days[i] = idx - i;
		}
	}

	return days;
}

int main(int, char**argv) {
	
	vector<int> T = {89,62,70,58,47,47,46,76,100,70};

	for (auto& e : T)
		cout << e << " ";
	cout << endl;
	for (auto& e : warmtemp(T))
		cout << e << " ";
	cout << endl;
	



    return 0;
}
