#!/bin/bash
ulimit -s unlimited
declare -a sizes=( 48 64 80 96 112 128 256 384 512 640 768 896 1024 2048 3072 4096 5120 6144 7168 8192 9216 )
tmp=32
for i in "${sizes[@]}"
do
        for f in *; do
                if [ -d "$f" ]; then
                        cd $f
                        echo "power_${f}_${tmp}"
                        perf stat -o ../power_${f}_${tmp}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./matMul
                        sed -i "s/$tmp/$i/g" ../../conf/settings.json
                        tmp=$i
                        cd ..
                fi
        done
done
