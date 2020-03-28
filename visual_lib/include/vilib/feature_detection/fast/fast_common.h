/*
 * FAST feature detector common definitions
 * fast_common.h
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

#pragma once

namespace vilib {

enum fast_score {
    SUM_OF_ABS_DIFF_ALL=0,  // OpenCV: https://docs.opencv.org/master/df/d0c/tutorial_py_fast.html
    SUM_OF_ABS_DIFF_ON_ARC, // Rosten 2006
    MAX_THRESHOLD           // Rosten 2008
};

} // namespace vilib
