CXX      ?= c++

UNAME_S  := $(shell uname -s)

# --- Bucket sieve: large-prime bucket scheduler (default: on) ---
# Override with: make BUCKET_SIEVE=0
BUCKET_SIEVE ?= 1

# --- CPU phase concurrency (default: off) ---
# Run LargeFree and LargeNonFree on two threads (fills idle cores; helps on
# high-core machines, costs peak memory). Enable with: make CPU_OVERLAP=1
CPU_OVERLAP ?= 0

# --- Base flags (all platforms) ---
CXXFLAGS := -O3 -std=c++17 \
            -I../sieve \
            -I../MertensHurst/src \
            -flto -fno-exceptions -fno-rtti \
            -frounding-math -finline-functions \
            -march=native -mtune=native \
            -fno-strict-overflow -DNDEBUG \
            -DUSE_BUCKET_SIEVE=$(BUCKET_SIEVE) \
            -DMHT_CPU_OVERLAP=$(CPU_OVERLAP)

# Extra defines, e.g. make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024 -DSIEVE_STRIDE_LOG=7"
CXXFLAGS += $(EXTRA_CXXFLAGS)
LDFLAGS  :=

# --- Platform-specific: OpenMP + library paths ---
ifeq ($(UNAME_S),Darwin)
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
    CXXFLAGS += -I$(BREW_PREFIX)/include \
                -I$(BREW_PREFIX)/opt/libomp/include \
                -Xpreprocessor -fopenmp
    LDFLAGS  += -L$(BREW_PREFIX)/opt/libomp/lib -lomp
else
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
endif

SRCDIR   = src
BUILDDIR = build
SIEVEDIR = ../sieve
SRCS     = $(SRCDIR)/MertensHT.cpp \
           $(SRCDIR)/main.cpp \
           $(SIEVEDIR)/SegmentedMobiusSieve.cpp \
           ../MertensHurst/src/MertensHurst.cpp

.PHONY: all clean

all: $(BUILDDIR)/mertens_ht

$(BUILDDIR)/mertens_ht: $(SRCS) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SIEVEDIR)/*.h)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR)
