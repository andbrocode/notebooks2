CC = gcc
CFLAGS = -g3 -ggdb -fPIC -fno-inline -Wall -Wextra

DEPS = \
        demodulate.o \
        hilbert.o \
        convolution.o 

demod.so: $(DEPS)
	$(CC) -shared $(CFLAGS) -lm -o demod.so $(DEPS)

install: demod.so
	cp demod.so ../lib/libdemod.so

clean:
	find ./ -name "*.o" -exec rm {} \;
	rm -f *.so

