CXX      ?= clang++
BREW     := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)

# Division-free mode: quotient cache on x86 (e.g. 2019 Mac Pro), direct
# division on ARM — same auto-detection as the other Makefiles.
UNAME_M  := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    DIVISION_FREE ?= 1
else
    DIVISION_FREE ?= 0
endif

BASEFLAGS := -O3 -std=c++17 \
             -flto -fno-exceptions -fno-rtti \
             -frounding-math -finline-functions \
             -march=native -mtune=native \
             -fno-strict-overflow -DNDEBUG \
             -DUSE_DIVISION_FREE=$(DIVISION_FREE) -DUSE_BUCKET_SIEVE=1

# Extra defines, e.g. make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024"
BASEFLAGS += $(EXTRA_CXXFLAGS)
OMPFLAGS  := -I$(BREW)/opt/libomp/include -Xpreprocessor -fopenmp
OMPLINK   := -L$(BREW)/opt/libomp/lib -lomp
MTLLINK   := -framework Metal -framework Foundation

# Standalone: MertensHurstGPU keeps its own copy of the MertensHurst core
# sources (src/), so it depends only on the shared sieve directory. The CPU
# MertensHurst build is GPU-free and is not referenced here.
MH       = src
SIEVE    = ../../sieve
BUILDDIR = build

all: $(BUILDDIR)/mertens_gpu

# MertensHurst with GPU S2 offload: this directory's own copy of the core
# sources (src/) plus the Metal engine, compiled with -DUSE_GPU_S2=1.
$(BUILDDIR)/gpu_s2.o: $(MH)/gpu_s2.mm $(MH)/gpu_s2.h
	@mkdir -p $(BUILDDIR)
	$(CXX) -O3 -std=c++17 -fobjc-arc -c $< -o $@

$(BUILDDIR)/mertens_gpu: $(BUILDDIR)/gpu_s2.o $(MH)/MertensHurst.cpp $(MH)/main.cpp \
                         $(SIEVE)/SegmentedMobiusSieve.cpp \
                         $(wildcard $(MH)/*.h) $(wildcard $(SIEVE)/*.h)
	$(CXX) $(BASEFLAGS) $(OMPFLAGS) -DUSE_GPU_S2=1 -I$(MH) -I$(SIEVE) \
	    $(MH)/MertensHurst.cpp $(SIEVE)/SegmentedMobiusSieve.cpp $(MH)/main.cpp \
	    $(BUILDDIR)/gpu_s2.o -o $@ $(OMPLINK) $(MTLLINK)

clean:
	rm -rf $(BUILDDIR)
