#!/bin/bash
source $PWD/progress_bar.sh
rm matGen
tmp=16
tasks_in_total="$(seq 32 16 16384 | wc -l)"
task=1
for i in $(seq 32 16 16384)
do
  	sed -i "s/const int N = ${tmp};/const int N = ${i};/g" matGen.cpp
        icpx matGen.cpp -o matGen -fopenmp -lpthread
        ./matGen
        tmp=$i
	show_progress $task $tasks_in_total
        task=$((task+1))
done