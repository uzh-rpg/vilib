/*
 * Argument holder
 * arguments.cpp
 *
 * Copyright (C) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
