#include <cstdio>

void show_bytes(char* start, int len) {
    for (int i = 0; i < len; i++) {
        printf(" %.2x", start[i]);
    }
    printf("\n");
}

void show_int(int x) {
    show_bytes((char*)&x, sizeof(int));
}

int main() {

    show_int(12345);


    return 0;
}
