#!/bin/bash
valgrind --tool=callgrind --callgrind-out-file='callgrind.%p' ./test_vilib
