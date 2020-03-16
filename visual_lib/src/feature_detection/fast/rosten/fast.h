#pragma once

namespace vilib {
namespace rosten {

typedef struct xy_tuple { int x, y; } xy;
typedef struct xys_tuple { int x, y, s; } xys;
typedef unsigned char byte;

xy* fast9_detect(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast10_detect(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast11_detect(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast12_detect(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);

template<bool use_new_score>
int* fast9_score(const byte* i, int stride, xy* corners, int num_corners, int b);

template<bool use_new_score>
int* fast10_score(const byte* i, int stride, xy* corners, int num_corners, int b);

template<bool use_new_score>
int* fast11_score(const byte* i, int stride, xy* corners, int num_corners, int b);

template<bool use_new_score>
int* fast12_score(const byte* i, int stride, xy* corners, int num_corners, int b);

template<bool use_new_score>
xys* fast9_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template<bool use_new_score>
xys* fast10_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template<bool use_new_score>
xys* fast11_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template<bool use_new_score>
xys* fast12_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);

xys* nonmax_suppression(const xy* corners, const int* scores, int num_corners, int* ret_num_nonmax);

} // namespace rosten
} // namespace vilib