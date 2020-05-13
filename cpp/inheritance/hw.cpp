#include <iostream>

using namespace std;

struct Common
{
	virtual double area() {return 0;};
};

struct Rectangular : public Common
{
	Rectangular(double length_, double width_): length(length_), width(width_) {}

	double area() { return length*width; }

	double length;
	double width;
};

struct Triangle : public Common
{
	Triangle(double base_, double height_): base(base_), height(height_) {}

	double area() { return 0.5*base*height; }
	double base;
	double height;
};

struct GetArea
{
	GetArea(Common* c): obj(c) {}

	Common* obj;
};

int main() {

	Rectangular r(3,5);
	Triangle t(2,8);

	GetArea g(&r), h(&t);
	cout << g.obj->area() << endl;
	cout << h.obj->area() << endl;

	return 0;
}
