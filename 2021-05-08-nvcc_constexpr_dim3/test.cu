#include <stdio.h>

template <int BLOCK_X, int BLOCK_Y>
__global__ void foo() {
    printf("hello\n");
}

int main(){
    constexpr dim3 block(32, 4, 1);
    foo<block.x, block.y><<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

