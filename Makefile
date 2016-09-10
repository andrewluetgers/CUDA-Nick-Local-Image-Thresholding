ARCH = sm_35 #Appropriate for Kepler GK110

Nick: Nick.cu Nick_kernel.cu Nick_gold.o 
	nvcc -arch=$(ARCH) -o Nick Nick.cu Nick_gold.o 
Nick_gold.o: Nick_gold.cpp
	nvcc -c -o Nick_gold.o Nick_gold.cpp
