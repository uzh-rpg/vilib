#!/bin/bash
EXECUTABLE_PATH="./test_vilib"
PROF_FILE_PATH="./test_vilib.gperf"
OUTPUT_FILE_PATH="./test_vilib.gperf.pdf"
google-pprof --pdf ${EXECUTABLE_PATH} ${PROF_FILE_PATH} > ${OUTPUT_FILE_PATH}
