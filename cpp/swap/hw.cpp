#include<iostream>
#include<vector>


int main(){


    std::vector<int> a={1, 2, 3};
    std::vector<int> b={4, 5, 6};

    int* pa = a.data();
    int* pb = b.data();

    std::swap(pa,pb);
    std::cout<<"after std::swap(pa,pb)\n";
    std::cout<<"a= "<<a[0]<<" "<<a[1]<<" "<<a[2]<<"\n";
    std::cout<<"b= "<<b[0]<<" "<<b[1]<<" "<<b[2]<<"\n";

    std::cout<<"pa= "<<pa[0]<<" "<<pa[1]<<" "<<pa[2]<<"\n";
    std::cout<<"pb= "<<pb[0]<<" "<<pb[1]<<" "<<pb[2]<<"\n";

    std::swap(a,b);

    std::cout<<"after std::swap(a,b)\n";
    std::cout<<"a= "<<a[0]<<" "<<a[1]<<" "<<a[2]<<"\n";
    std::cout<<"b= "<<b[0]<<" "<<b[1]<<" "<<b[2]<<"\n";

    std::cout<<"pa= "<<pa[0]<<" "<<pa[1]<<" "<<pa[2]<<"\n";
    std::cout<<"pb= "<<pb[0]<<" "<<pb[1]<<" "<<pb[2]<<"\n";
}
