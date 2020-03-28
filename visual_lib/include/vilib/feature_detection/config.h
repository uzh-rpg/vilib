/*
 * Base class for GPU feature detectors
 * config.h
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

// Non-maximum suppression size
#define DETECTOR_BASE_NMS_SIZE                          3

#if (DETECTOR_BASE_NMS_SIZE%2 == 0)
#error "Error: the non-maximum suppression size should be an odd number"
#endif /* (DETECTOR_BASE_NMS_SIZE%2 == 0) */

} // namespace vilib
