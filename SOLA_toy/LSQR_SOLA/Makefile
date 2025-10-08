#== BEGIN USER SETTINGS =============================================
# Executable for LSQR SOLA
EXEC1 = LSQR_inversion_for_SOLA_N

# Executable for Ak / R computation 
EXEC2 = Compute_AveragingKernels_R_for_SOLA

# C compiler
CC    =  gcc

# Flags for debugging or for maximum performance, comment as necessary
# Note that some options might be compiler dependent
CFLAGS =  -lm 
#== END USER SETTINGS =============================================

SRCS1=fzsparse.c catch_sig.c lsqr_wrapper.c matrice.c lsqr.c $(EXEC1).c
OBJS1=$(SRCS1:%.c=%.o)

SRCS2=fzsparse.c catch_sig.c lsqr_wrapper.c matrice.c lsqr.c $(EXEC2).c
OBJS2=$(SRCS2:%.c=%.o)

all: $(EXEC1) $(EXEC2)

$(EXEC1): $(OBJS1)
	${CC} $(CFLAGS) $(OBJS1) -o $@

$(EXEC2): $(OBJS2)
	${CC} $(CFLAGS) $(OBJS2) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $(@F:.o=.c) -o $@

clean:
	rm -f $(CMD) *.o core
