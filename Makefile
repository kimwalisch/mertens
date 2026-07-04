.PHONY: all hurst ht sieve clean

all: hurst ht

hurst:
	$(MAKE) -C MertensHurst

ht:
	$(MAKE) -C MertensHT

sieve:
	$(MAKE) -C sieve

clean:
	$(MAKE) -C MertensHurst clean
	$(MAKE) -C MertensHT clean
	$(MAKE) -C sieve clean
