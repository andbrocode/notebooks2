#include <stdio.h>

void say_hello(int times)
{
    for (int i = 0; i < times; ++i)
    {
        printf("%d. Hello, Python. I am C!\n", i+1);
    }
}
