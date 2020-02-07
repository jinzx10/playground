#include <iostream>

template <typename T>
struct Base
{
	void interface() {
        static_cast<T*>(this)->print();
    }

};

struct Cat: public Base<Cat>
{
	int i;
	void print() { std::cout << "cat" << std::endl; }
};

struct Dog: public Base<Dog>
{
	void print() { std::cout << "dog" << std::endl; }
};

struct Man
{
	void print() { std::cout << "man" << std::endl; }
	int man;
};

struct Tq : public Man
{
	void print() { std::cout << "tq" << std::endl; }
	Cat* tq_s_cat;
};

int main() {
	Base<Cat> b;
	b.interface();

	Base<Dog> c;
	c.interface();

	Cat cat;

	//Cat* dd = &b;
	Cat* cc = static_cast<Cat*>(&b);

	Man* pman = new Tq;
	pman->print();

	Man man;
	Tq tq;

	Tq* ptq = static_cast<Tq*>(&man);
	std::cout << ptq->tq_s_cat->i << std::endl; 
	//ptq->Man::print();
	//ptq->print();

	return 0;
}
