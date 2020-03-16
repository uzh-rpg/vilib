/*
 * FAST feature detector common definitions
 * fast_common.h
 */

#pragma once

namespace vilib {

enum fast_score {
    SUM_OF_ABS_DIFF_ALL=0,  // OpenCV: https://docs.opencv.org/master/df/d0c/tutorial_py_fast.html
    SUM_OF_ABS_DIFF_ON_ARC, // Rosten 2006
    MAX_THRESHOLD           // Rosten 2008
};

} // namespace vilib
