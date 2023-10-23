------------------------------------------------------------------------------------------------------------
|   CPU
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

export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_DCM,PAPI_FMA_INS,PAPI_FP_INS,PAPI_L1_DCR,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L1_DCW,PAPI_L2_DCW,PAPI_L3_DCW,perf::TASK-CLOCK,PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_FP_INSPAPI_FP_OPS"
export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_L1_DCM,PAPI_REF_CYC,PAPI_VEC_DP,PAPI_VEC_SP,PAPI_DP_OPS,PAPI_SP_OPS"
export PAPI_EVENTS="rapl:::PACKAGE_ENERGY:PACKAGE0,rapl:::PACKAGE_ENERGY:PACKAGE1,PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_DCM,PAPI_FMA_INS,PAPI_FP_INS,PAPI_L1_DCR,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L1_DCW,PAPI_L2_DCW,PAPI_L3_DCW,PAPI_FP_OPS,PAPI_SP_OPS,PAPI_VEC_SP"

------------------------------------------------------------------------------------------------------------
|   GPU
------------------------------------------------------------------------------------------------------------
export PAPI_CUDA_ROOT=/usr/local/cuda-10.1
papi_native_avail | grep nvml

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