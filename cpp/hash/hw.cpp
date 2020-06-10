#include <iostream>
#include <vector>


using namespace std;


class IntHash
{
	public:
		static const unsigned int sz_max = 1e6;
		unsigned int vals[sz_max];

		unsigned int hash_func(int key) {
			int res = key % sz_max;
			return res < 0 ? res + sz_max : res;
		}

		void add(int key, unsigned int val) {
			vals[hash_func(key)] = val;
		}

		int get(int key) {
			return vals[hash_func(key)];
		}
};

int main() {

	IntHash ih;

	ih.add(32, 2);
	ih.add(34, 4);
	ih.add(35, 7);
	ih.add(36, 8);
	ih.add(40, 9);

	cout << ih.get(40) << endl;
	cout << ih.get(36) << endl;
	cout << ih.get(99) << endl;

	return 0;
}
