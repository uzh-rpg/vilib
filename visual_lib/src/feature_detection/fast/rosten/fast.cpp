#include <stdlib.h>
#include "fast.h"

namespace vilib {
namespace rosten {

template <bool use_new_score>
xys* fast9_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners)
{
	xy* corners;
	int num_corners;
	int* scores;
	xys* nonmax;

	corners = fast9_detect(im, xsize, ysize, stride, b, &num_corners);
	scores = fast9_score<use_new_score>(im, stride, corners, num_corners, b);
	nonmax = nonmax_suppression(corners, scores, num_corners, ret_num_corners);

	free(corners);
	free(scores);

	return nonmax;
}

template <bool use_new_score>
xys* fast10_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners)
{
	xy* corners;
	int num_corners;
	int* scores;
	xys* nonmax;

	corners = fast10_detect(im, xsize, ysize, stride, b, &num_corners);
	scores = fast10_score<use_new_score>(im, stride, corners, num_corners, b);
	nonmax = nonmax_suppression(corners, scores, num_corners, ret_num_corners);

	free(corners);
	free(scores);

	return nonmax;
}

template <bool use_new_score>
xys* fast11_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners)
{
	xy* corners;
	int num_corners;
	int* scores;
	xys* nonmax;

	corners = fast11_detect(im, xsize, ysize, stride, b, &num_corners);
	scores = fast11_score<use_new_score>(im, stride, corners, num_corners, b);
	nonmax = nonmax_suppression(corners, scores, num_corners, ret_num_corners);

	free(corners);
	free(scores);

	return nonmax;
}

template <bool use_new_score>
xys* fast12_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners)
{
	xy* corners;
	int num_corners;
	int* scores;
	xys* nonmax;

	corners = fast12_detect(im, xsize, ysize, stride, b, &num_corners);
	scores = fast12_score<use_new_score>(im, stride, corners, num_corners, b);
	nonmax = nonmax_suppression(corners, scores, num_corners, ret_num_corners);

	free(corners);
	free(scores);

	return nonmax;
}

// Explicit instantiations
template xys* fast9_detect_nonmax<false>(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template xys* fast9_detect_nonmax<true>(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template xys* fast10_detect_nonmax<false>(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template xys* fast10_detect_nonmax<true>(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template xys* fast11_detect_nonmax<false>(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template xys* fast11_detect_nonmax<true>(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template xys* fast12_detect_nonmax<false>(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
template xys* fast12_detect_nonmax<true>(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);

} // namespace rosten
} // namespace vilib