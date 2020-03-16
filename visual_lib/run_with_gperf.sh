#!/bin/bash
if [ -f /usr/lib/libprofiler.so ]; then
	PROF_LIBRARY_PATH="/usr/lib/libprofiler.so";
elif [ -f /usr/lib/x86_64-linux-gnu/libprofiler.so ]; then
	PROF_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/libprofiler.so";
else
	echo "GPerf profiling library wasn't found"
	exit
fi
EXECUTABLE_PATH="./test_vilib"
PROF_FILE_PATH="./test_vilib.gperf"
env CPUPROFILE_FREQUENCY=200 CPUPROFILE=${PROF_FILE_PATH} LD_PRELOAD=${PROF_LIBRARY_PATH} ${EXECUTABLE_PATH}
