.PHONY: all build test selftest demo clean

CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -fPIC -Wall -Wextra

all: build test selftest

build: libprod_gate_simd.so

libprod_gate_simd.so: prod_gate_native_simd.c
	$(CC) $(CFLAGS) -shared -o libprod_gate_simd.so prod_gate_native_simd.c -lm

test:
	python3 -m unittest discover -v -s . -p "test_*.py"

selftest:
	python3 electric_vision_cli.py --no-color --plain --mode focus selftest
	python3 electric_audio_cli.py --no-color --plain --mode focus selftest

demo:
	python3 examples/connect_anything_demo.py

clean:
	rm -rf __pycache__ .pytest_cache prod_gate_verification_report.json
