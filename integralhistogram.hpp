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

#ifndef INTEGRALHISTOGRAM_HPP
#define INTEGRALHISTOGRAM_HPP

#include <opencv2/opencv.hpp>

using namespace cv;
  
template<typename imType, typename binType>
class IntegralHistogram
{
public:
  template<typename simType>
  struct Compare
  {
      typedef simType (*f)(binType *h1, binType *h2, int len);
  };
  
private:
  Size mDim;
  int mNChannels;
  int mNBins;
  imType mMaxVal;

public:
  IntegralHistogram(Size dim, int nchannels, int nbins,
                    imType maxval = std::numeric_limits<imType>::max());
  void integralHistogram(InputArray image, std::vector<binType> &hist);
  void integralHistogramVM(InputArray val, InputArray mag,
                           std::vector<binType>& hist);
  void integralHistogramJoint( InputArray val, InputArray mag,
                               std::vector<binType>& hist, int nmag);
  
  template<typename simType>
  void compare(std::vector<binType> &h1, std::vector<binType> &h2,
               Size size, OutputArray out, 
               typename Compare<simType>::f cmp);
  void regionHistogram(const std::vector<binType> &integral, 
                       const Rect &region,
                       std::vector<binType> &out);

  
private:
  void calcHist(Mat_<uchar> image, Size size, int *desc);
  void wavefrontScan(Mat_<imType> image, binType *hist);
  void wavefrontScanVM(Mat_<imType> val, Mat_<imType> mag,
                       binType* hist);
  void wavefrontScanJoint(Mat_<imType> val, Mat_<imType> mag,
                          binType* hist, int nmag);
  
  template<typename simType>
  void compSingle(const binType *h1, const binType *h2,
                  Size size, Mat_<simType> out, 
                  typename Compare<simType>::f cmp
                 );
  inline void sumHist(binType *hist1, binType *hist2,
                      binType *hist3, binType *out);
  inline void regionHist(const binType *hist00, const binType *hist01, 
                         const binType *hist10, const binType *hist11,
                         binType *out);
  template<typename simType>
  inline simType compHist(binType *hist1, binType *hist2,
                          const Size &size);
  inline void histMatch(const binType *temp, const binType *hist, binType *out);
};

#include "integralhistogram.cpp"
#endif // INTEGRALHISTOGRAM_HPP

