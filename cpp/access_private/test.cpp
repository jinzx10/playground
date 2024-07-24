#include "./access_private.hpp"
#include <cassert>

class A {
  int m_i = 3;
  int m_f(int p) { return 14 * p; }
};

ACCESS_PRIVATE_FIELD(A, int, m_i)

void foo() {
  A a;
  auto &i = access_private::m_i(a);
  assert(i == 3);
}

ACCESS_PRIVATE_FUN(A, int(int), m_f)

void bar() {
  A a;
  int p = 3;
  auto res = call_private::m_f(a, p);
  assert(res == 42);
}

int main() {

    foo();
    bar();

    return 0;
}
