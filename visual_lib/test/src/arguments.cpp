/*
 * Argument holder
 * arguments.cpp
 */

#include <iostream>
#include <string.h>
#include <libgen.h>
#include "test/arguments.h"

static int app_argument_count;
static char ** app_arguments;
static std::string app_folder_path;

void init_arguments(int argc, char ** argv) {
  app_argument_count = argc;
  app_arguments = argv;
  // extract the executable's folder path
  char * path = realpath(dirname(app_arguments[0]),NULL);
  app_folder_path = path;
  free(path);
}

std::string get_executable_folder_path(void) {
  return app_folder_path;
}
