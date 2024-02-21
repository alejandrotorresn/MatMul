sh -c "echo 2 > /proc/sys/kernel/perf_event_paranoid"

------------------------------------------------------------------------------------------------------------
|   CPU - Compile
------------------------------------------------------------------------------------------------------------
. /opt/intel/oneapi/setvars.sh
icpx -fsycl -fsycl-targets=x86_64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -m64 -ljsoncpp -I"${MKLROOT}/include" -I /opt/papi/include ../handle_error.c /opt/papi/lib/libpapi.a -O3 matMul.cpp -o matMul

icpx -fsycl -fsycl-targets=x86_64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl  -DMKL_ILP64  -m64 -qopenmp -ljsoncpp -I"${MKLROOT}/include" -I /opt/papi/include ../handle_error.c /opt/papi/lib/libpapi.a -O3 -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw -mavx512dq matMul.cpp -o matMul

icpx -fsycl -fsycl-targets=x86_64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -m64 qopenmp -ljsoncpp -I"${MKLROOT}/include" -I /opt/papi/include ../handle_error.c /opt/papi/lib/libpapi.a -O3 matMul_cpu.cpp -o matMul_cpu
------------------------------------------------------------------------------------------------------------
|   CPU
------------------------------------------------------------------------------------------------------------
export PAPI_MULTIPLEX=1
export LIBPFM_FORCE_PMU=core
export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,rapl:::DRAM_ENERGY:PACKAGE0,rapl:::DRAM_ENERGY:PACKAGE1,PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_DCM,PAPI_FMA_INS,PAPI_FP_INS,PAPI_L1_DCR,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L1_DCW,PAPI_L2_DCW,PAPI_L3_DCW,perf::TASK-CLOCK,PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_FP_INSPAPI_FP_OPS"

------------------------------------------------------------------------------------------------------------
|   GPU - Compile
------------------------------------------------------------------------------------------------------------
nvcc -lcublas -ljsoncpp matMul.cu ../include/cuda_mm.cu -o matMul

icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -m64 -ljsoncpp -I"${MKLROOT}/include" -I /opt/papi/include /opt/papi/lib/libpapi.a -O3 ../handle_error.c matMul.cpp -o matMul

------------------------------------------------------------------------------------------------------------
|   GPU
------------------------------------------------------------------------------------------------------------
export PAPI_CUDA_ROOT=/usr/local/cuda
papi_native_avail | grep nvml
export PAPI_EVENTS="nvml:::Tesla_V100-PCIE-32GB:device_0:power,nvml:::Tesla_V100-PCIE-32GB:device_0:gpu_utilization,nvml:::Tesla_V100-PCIE-32GB:device_0:memory_utilization"

export PAPI_EVENTS="nvml:::NVIDIA_A100-PCIE-40GB:device_0:power,nvml:::NVIDIA_A100-PCIE-40GB:device_0:gpu_utilization,nvml:::NVIDIA_A100-PCIE-40GB:device_0:memory_utilization"

------------------------------------------------------------------------------------------------------------
|   CPU - Events
------------------------------------------------------------------------------------------------------------
PAPI_L1_DCM  0x80000000  Yes   No   Level 1 data cache misses
PAPI_L2_DCM  0x80000002  Yes   Yes  Level 2 data cache misses
PAPI_L3_DCM  0x80000004  No    No   Level 3 data cache misses
PAPI_FMA_INS 0x80000030  No    No   FMA instructions completed
PAPI_FP_INS  0x80000034  Yes   No   Floating point instructions
PAPI_L1_DCR  0x80000043  No    No   Level 1 data cache reads
PAPI_L2_DCR  0x80000044  Yes   No   Level 2 data cache reads
PAPI_L3_DCR  0x80000045  No    No   Level 3 data cache reads
PAPI_L1_DCW  0x80000046  No    No   Level 1 data cache writes
PAPI_L2_DCW  0x80000047  Yes   No   Level 2 data cache writes
PAPI_L3_DCW  0x80000048  No    No   Level 3 data cache writes
PAPI_FP_OPS  0x80000066  Yes   No   Floating point operations
PAPI_SP_OPS  0x80000067  Yes   Yes  Floating point operations; optimized to count scaled single precision vector operations
PAPI_VEC_SP  0x80000069  Yes   No   Single precision vector/SIMD instructions

rapl:::PACKAGE_ENERGY:PACKAGE0
rapl:::PACKAGE_ENERGY:PACKAGE1
rapl:::DRAM_ENERGY:PACKAGE0
rapl:::DRAM_ENERGY:PACKAGE1

CACHES

export LIBPFM_FORCE_PMU=icx
"PAPI_L1_DCM":"605",
"PAPI_L2_DCR":"454",
"PAPI_L3_DCR":"215"

export LIBPFM_FORCE_PMU=core
"PAPI_L2_DCM":"242"

export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,rapl:::DRAM_ENERGY:PACKAGE0,rapl:::DRAM_ENERGY:PACKAGE1,PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_DCM,PAPI_FMA_INS,PAPI_FP_INS,PAPI_L1_DCR,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L1_DCW,PAPI_L2_DCW,PAPI_L3_DCW,perf::TASK-CLOCK,PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_FP_INSPAPI_FP_OPS"
export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_L1_DCM,PAPI_REF_CYC,PAPI_VEC_DP,PAPI_VEC_SP,PAPI_DP_OPS,PAPI_SP_OPS"
export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_DCM,PAPI_FMA_INS,PAPI_FP_INS,PAPI_L1_DCR,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L1_DCW,PAPI_L2_DCW,PAPI_L3_DCW,PAPI_FP_OPS,PAPI_SP_OPS,PAPI_VEC_SP"

export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,amd64_rapl::RAPL_ENERGY_PKG,rapl:::PACKAGE_ENERGY_CNT:PACKAGE0,rapl:::PACKAGE_ENERGY_CNT:PACKAGE1,rapl:::PP0_ENERGY_CNT:PACKAGE0,rapl:::PP0_ENERGY_CNT:PACKAGE1,PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_DCM,PAPI_FMA_INS,PAPI_FP_INS,PAPI_L1_DCR,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L1_DCW,PAPI_L2_DCW,PAPI_L3_DCW,perf::TASK-CLOCK,PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_FP_INSPAPI_FP_OPS"

export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_L1_DCM,PAPI_REF_CYC,PAPI_VEC_DP,PAPI_VEC_SP,PAPI_DP_OPS,PAPI_SP_OPS"
export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_DCM,PAPI_FMA_INS,PAPI_FP_INS,PAPI_L1_DCR,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L1_DCW,PAPI_L2_DCW,PAPI_L3_DCW,PAPI_FP_OPS,PAPI_SP_OPS,PAPI_VEC_SP"



------------------------------------------------------------------------------------------------------------
|   GPU - Events
------------------------------------------------------------------------------------------------------------

Architecture V100    
nvml:::Tesla_V100-PCIE-32GB:device_0:power
nvml:::Tesla_V100-PCIE-32GB:device_0:gpu_utilization
nvml:::Tesla_V100-PCIE-32GB:device_0:memory_utilization

nvml:::Tesla_V100-PCIE-32GB:device_0:l1_single_ecc_errors
nvml:::Tesla_V100-PCIE-32GB:device_0:l2_single_ecc_errors
nvml:::Tesla_V100-PCIE-32GB:device_0:memory_single_ecc_errors
nvml:::Tesla_V100-PCIE-32GB:device_0:regfile_single_ecc_errors
nvml:::Tesla_V100-PCIE-32GB:device_0:1l_double_ecc_errors
nvml:::Tesla_V100-PCIE-32GB:device_0:l2_double_ecc_errors
nvml:::Tesla_V100-PCIE-32GB:device_0:memory_double_ecc_errors
nvml:::Tesla_V100-PCIE-32GB:device_0:regfile_double_ecc_errors

export PAPI_EVENTS="nvml:::Tesla_V100-PCIE-32GB:device_0:power,nvml:::Tesla_V100-PCIE-32GB:device_0:gpu_utilization,nvml:::Tesla_V100-PCIE-32GB:device_0:memory_utilization"





Native Events in Component: nvml
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:graphics_clock                         |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:sm_clock                               |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:memory_clock                           |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:l1_single_ecc_errors                   |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:l2_single_ecc_errors                   |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:memory_single_ecc_errors               |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:regfile_single_ecc_errors              |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:1l_double_ecc_errors                   |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:l2_double_ecc_errors                   |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:memory_double_ecc_errors               |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:regfile_double_ecc_errors              |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:graphics_max_clock                     |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:sm_max_clock                           |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:memory_max_clock                       |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:total_memory                           |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:unallocated_memory                     |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:allocated_memory                       |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:pstate                                 |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:power                                  |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:temperature                            |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:total_ecc_errors                       |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:total_ecc_errors                       |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:gpu_utilization                        |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:memory_utilization                     |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:power_management_limit                 |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:power_management_limit_constraint_min    |
| nvml:::NVIDIA_A100-PCIE-40GB:device_0:power_management_limit_constraint_max    |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:graphics_clock                         |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:sm_clock                               |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:memory_clock                           |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:l1_single_ecc_errors                   |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:l2_single_ecc_errors                   |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:memory_single_ecc_errors               |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:regfile_single_ecc_errors              |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:1l_double_ecc_errors                   |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:l2_double_ecc_errors                   |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:memory_double_ecc_errors               |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:regfile_double_ecc_errors              |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:graphics_max_clock                     |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:sm_max_clock                           |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:memory_max_clock                       |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:total_memory                           |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:unallocated_memory                     |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:allocated_memory                       |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:pstate                                 |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:power                                  |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:temperature                            |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:total_ecc_errors                       |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:total_ecc_errors                       |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:gpu_utilization                        |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:memory_utilization                     |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:power_management_limit                 |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:power_management_limit_constraint_min    |
| nvml:::NVIDIA_A100-PCIE-40GB:device_1:power_management_limit_constraint_max    |





------------------------------------------------------------------------------------------------------------
|   AMD
------------------------------------------------------------------------------------------------------------



| amd64_rapl::RAPL_ENERGY_PKG                                                  |
| rapl:::PACKAGE_ENERGY_CNT:PACKAGE0                                           |
| rapl:::PACKAGE_ENERGY_CNT:PACKAGE1                                           |
| rapl:::PP0_ENERGY_CNT:PACKAGE0                                               |
| rapl:::PP0_ENERGY_CNT:PACKAGE1                                               |
| rapl:::PACKAGE_ENERGY:PACKAGE0                                               |
| rapl:::PACKAGE_ENERGY:PACKAGE1                                               |
| rapl:::PP0_ENERGY:PACKAGE0                                                   |
| rapl:::PP0_ENERGY:PACKAGE1                                                   |


# ------------------------------------------------------------------------------------------------------------
# Perf Power Consumption
# ------------------------------------------------------------------------------------------------------------
icpx -fsycl -fsycl-targets=x86_64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl  -DMKL_ILP64  -m64 -qopenmp -ljsoncpp -I"${MKLROOT}/include" -O3 -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw -mavx512dq matMul.cpp -o matMul

nano ../../conf/settings.json
perf stat -o power_avx2_32.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./matMul
sed -i 's/{"N":32}/{"N":48}/g' ../../conf/settings.json
perf stat -o power_avx2_48.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./matMul
sed -i 's/{"N":96}/{"N":112}/g' ../../conf/settings.json
perf stat -o power_avx2_64.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./matMul
perf stat -o power_avx2_80.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./matMul
sed -i 's/{"N":96}/{"N":112}/g' ../../conf/settings.json