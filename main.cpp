/*
 * Copyright 2017 Jan Dorazil <deu439@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "integralhistogram.hpp"

#define N_CHANNELS  3
#define N_BINS      20


float compare_X2(uint16_t *h1, uint16_t *h2, int len)
{
  float sim = 0;
  float val;
  for( int i = 0; i < len; i++ ){
    val = h1[i] - h2[i];
    if(val != 0)
      sim += (val*val) / (h1[i] + h2[i]);
  }
  return sim;
}

int main(int argc, char **argv) {
  Mat A = imread("ima.tiff", IMREAD_COLOR);
  Mat B = imread("imb.tiff", IMREAD_COLOR);
  
  // Instantiate IntegralHistogram with uchar image type
  // and uint16_t bin type
  // You have to supply the image size and maximum pixel value
  IntegralHistogram<uchar, uint16_t> hist(
    A.size(), N_CHANNELS, N_BINS, std::numeric_limits<uchar>::max()
  );
  
  // Calculate integral histogram for each image
  std::vector<uint16_t> histA, histB;
  hist.integralHistogram(A, histA);
  hist.integralHistogram(B, histB);
  
  // Now we can calculate histogram for any window position and size very fast 
  // in O(N_BINS)
  std::vector<uint16_t> out;
  hist.regionHistogram(histA, Rect(0, 0, 100, 100), out);
  
  // Print out histogram values
  // For color images the histograms for each color channel are stacked 
  // one after the other. This way we get N_BINS * N_CHANNELS values.
  std::vector<uint16_t>::const_iterator i = out.begin();
  for(i; i != out.end(); ++i)
    std::cout << *i << ", ";
  std::cout << std::endl;
  
  // Compare the images using histograms in sliding window.
  // The output image type is float
  // We've got some interesting results with LTP of the actual images
  // We implemented the X^2 distance measure (function compHist()).
  Mat sim;
  hist.compare<float>(histA, histB, Size(20, 20), sim, compare_X2);
  
  // Show the similarity image
  normalize(sim, sim, 0, 1, NORM_MINMAX);
  imshow("Similarity", sim);
  waitKey();
  
  return 0;
}
