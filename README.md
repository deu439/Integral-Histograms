Integral histograms

This is implementation of integral histograms as described in [1]. It can be 
used for object detection and localization. We used it for change detection 
with LTP textural descriptors. Because of this our implementation is in some 
aspects specialized for this problem.

We implemented three specific ways of calculating the integral histogram:
- integralHistogram(): Calculates the integral histogram on one input image
- integralHistogramVM(): Calculates the integral histogram on two images. The 
  pixel intensity of Val image determines the bin index and the pixel intensity 
  of Mag determines the increment in magnitude of this bin.
- integralHistogramJoint(): Calculates the integral histogram on two images. 
  The resulting histogram is two dimensional and is build in the sense of Joint 
  PDF.

Once your integral histogram is calculated, you can get histograms of any 
rectangular portion of the original image using function regionHistogram(). 
This histogram is calculated in linear time with respect to number of histogram 
bins.

Using the compare() function you can compare two images using histograms 
calculated in a sliding window of arbitrary size. For reasonably sized images 
this can be done in realtime.

See main.cpp for example usage.

[1] PORIKLI, F. Integral histogram: a fast way to extract histograms in 
Cartesian spaces. In: 2005 IEEE Computer Society Conference on Computer Vision 
and Pattern Recognition (CVPR'05). IEEE, 2005, 829-836 vol. 1. DOI: 
10.1109/CVPR.2005.188. ISBN 0-7695-2372-2.
