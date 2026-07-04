# ============================================================================
# MertensHTGPU — standalone GPU-experimental copy of MertensHT.
#
# Standalone: keeps its own copy of the MertensHT core sources (src/). It
# refers read-only to the shared ../../sieve and ../../MertensHurst/src (for
# BruteM); it never modifies them. NOT wired into the top-level Makefile.
#
# GPU path is Metal compute (portable across Apple Silicon and the 2019 Mac
# Pro AMD GPUs; multi-device). Build with GPU=1 to compile the Metal engine;
# GPU=0 (default for now) builds the CPU baseline so we always have a
# validated reference.
# ============================================================================

CXX      ?= clang++
UNAME_S  := $(shell uname -s)
UNAME_M  := $(shell uname -m)

# Homebrew prefix. Do NOT trust the `brew` binary — on a wrong-arch install
# (e.g. an arm64 Homebrew on an Intel Mac) it crashes with "Illegal instruction"
# and poisons the include/lib paths. Pick by CPU architecture instead; override
# with `make BREW=/your/prefix` if your layout differs.
ifeq ($(UNAME_M),arm64)
    BREW ?= /opt/homebrew
else
    BREW ?= /usr/local
endif

# Toggle the GPU (Metal) build path.
GPU          ?= 0
BUCKET_SIEVE ?= 1

SRCDIR   = src
SIEVE    = ../../sieve
MH       = ../../MertensHurst/src
BUILDDIR = build

CXXFLAGS := -O3 -std=c++17 \
            -I$(SRCDIR) -I$(SIEVE) -I$(MH) \
            -flto -fno-exceptions -fno-rtti \
            -frounding-math -finline-functions \
            -march=native -mtune=native \
            -fno-strict-overflow -DNDEBUG \
            -DUSE_BUCKET_SIEVE=$(BUCKET_SIEVE)

# Extra defines, e.g. make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024"
CXXFLAGS += $(EXTRA_CXXFLAGS)
LDFLAGS  :=

ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -I$(BREW)/include \
                -I$(BREW)/opt/libomp/include \
                -Xpreprocessor -fopenmp
    LDFLAGS  += -L$(BREW)/opt/libomp/lib -lomp
else
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
endif

# Core sources (CPU). The GPU engine object is added below when GPU=1.
SRCS = $(SRCDIR)/MertensHT.cpp \
       $(SRCDIR)/main.cpp \
       $(SIEVE)/SegmentedMobiusSieve.cpp \
       $(MH)/MertensHurst.cpp
OBJS_EXTRA =

ifeq ($(GPU),1)
    CXXFLAGS   += -DUSE_GPU=1
    LDFLAGS    += -framework Metal -framework Foundation
    OBJS_EXTRA += $(BUILDDIR)/gpu_engine.o
endif

.PHONY: all clean
all: $(BUILDDIR)/mertens_ht_gpu

# Metal engine (Objective-C++); only built when GPU=1.
$(BUILDDIR)/gpu_engine.o: $(SRCDIR)/gpu_engine.mm $(wildcard $(SRCDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) -O3 -std=c++17 -fobjc-arc -I$(SRCDIR) -I$(SIEVE) -I$(MH) -DUSE_GPU=1 -c $< -o $@

$(BUILDDIR)/mertens_ht_gpu: $(SRCS) $(OBJS_EXTRA) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVE)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(SRCS) $(OBJS_EXTRA) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR)
