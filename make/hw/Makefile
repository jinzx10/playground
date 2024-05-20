ROOTDIR = $(shell pwd)
SDIR = ${ROOTDIR}/src
IDIR = ${ROOTDIR}/include

ODIR = ${ROOTDIR}/obj
LDIR = ${ROOTDIR}/lib
BDIR = ${ROOTDIR}/bin

#VPATH = ${SDIR}:${IDIR}
vpath %.h ${IDIR}
vpath %.cpp ${SDIR}

CC = $(CXX)
CPPFLAGS = -I${IDIR}

OBJ0 = main fib
STATIC_LIB0 = fact dfact
SHARED_LIB0 = tq

# -Wl,-rpath=dir (Linux)
# -Wl,-rpath dir (MacOS)
LDFLAGS = -L${LDIR} -Wl,-rpath ${LDIR}
LDFLAGS += $(foreach LIB, ${STATIC_LIB0} ${SHARED_LIB0}, -l${LIB})

OBJS = $(patsubst %, ${ODIR}/%.o, ${OBJ0})
STATIC_LIBS = $(patsubst %, ${LDIR}/lib%.a, ${STATIC_LIB0})
SHARED_LIBS = $(patsubst %, ${LDIR}/lib%.so, ${SHARED_LIB0})

${BDIR}/main 	: ${OBJS} ${STATIC_LIBS} ${SHARED_LIBS} | ${BDIR}
	${CC} ${OBJS} -o $@ ${CPPFLAGS} ${LDFLAGS}

${ODIR}/main.o 	: main.cpp | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${ODIR}/%.o		: %.cpp %.h | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${LDIR}/lib%.a  : ${ODIR}/%.o | ${LDIR}
	ar -rcs $@ $<

${LDIR}/lib%.so : ${ODIR}/%.o | ${LDIR}
	${CC} -shared -o $@ $<

${BDIR} ${ODIR} ${LDIR} :
	mkdir -p $@

#.PRECIOUS: ${ODIR}/%.o
.PHONY: clean
clean:
	rm -rf ${ODIR} ${BDIR} ${LDIR}
