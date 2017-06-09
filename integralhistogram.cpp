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

#include <opencv2/opencv.hpp>

#include "integralhistogram.hpp"

using namespace cv;

template<typename imType, typename binType>
IntegralHistogram<imType, binType>::IntegralHistogram(
  Size dim, int nchannels, int nbins, imType maxval
)
  : mDim(dim), mNChannels(nchannels), mNBins(nbins), mMaxVal(maxval)
{
}

template<typename imType, typename binType>
void IntegralHistogram<imType, binType>::integralHistogram(
  InputArray image, std::vector<binType>& hist
)
{
  CV_Assert(image.channels() == mNChannels);
  
  int rows = mDim.height;
  int cols = mDim.width;
  int hist_rows = rows + 1;
  int hist_cols = cols + 1;
  int hist_len = hist_rows * hist_cols * mNBins;
  int row_len = hist_cols * mNBins;
  
  // Allocate vector for integral histogram added row and column
  hist.resize(hist_len * mNChannels);
  
  // Get data array
  binType *p_hist = hist.data();
  
  // Zero out the additional row
  for( int x = 0; x < row_len; x++ ){
    p_hist[x] = 0;
  }
  
  // Zero out the additional column
  for( int y = 1; y < hist_rows; y++  ){
    for( int x = 0; x < mNBins; x++ ){
      p_hist[y * row_len + x] = 0;
    }
  }

  // Split each channel
  std::vector<Mat> channels;
  if( mNChannels != 1 )
    split(image, channels);
  else
    channels.push_back(image.getMat());
    
  // Calculate integral histogram for each channel
  for( int i = 0; i < mNChannels; i++ ){
    wavefrontScan(channels[i], p_hist);
    p_hist += hist_len; // Skip onto next channel
  }
}

template<typename imType, typename binType>
void IntegralHistogram<imType, binType>::integralHistogramVM(
  InputArray val, InputArray mag, std::vector<binType>& hist
)
/* Value-magnitude version of integral histogram */
{
  CV_Assert( val.channels() == mNChannels );
  CV_Assert( mag.depth() == val.depth() );
  
  int rows = mDim.height;
  int cols = mDim.width;
  int hist_rows = rows + 1;
  int hist_cols = cols + 1;
  int hist_len = hist_rows * hist_cols * mNBins;
  int row_len = hist_cols * mNBins;
  
  // Allocate vector for integral histogram added row and column
  hist.resize(hist_len * mNChannels);
  
  // Get data array
  binType *p_hist = hist.data();
  
  // Zero out the additional row
  for( int x = 0; x < row_len; x++ ){
    p_hist[x] = 0;
  }
  
  // Zero out the additional column
  for( int y = 1; y < hist_rows; y++  ){
    for( int x = 0; x < mNBins; x++ ){
      p_hist[y * row_len + x] = 0;
    }
  }

  // Split each channel
  std::vector<Mat> channelsV, channelsM;
  if( mNChannels != 1 ){
    split(val, channelsV);
    split(mag, channelsM);
  } else {
    channelsV.push_back(val.getMat());
    channelsM.push_back(mag.getMat());
  }
    
  // Calculate integral histogram for each channel
  for( int i = 0; i < mNChannels; i++ ){
    wavefrontScanVM(channelsV[i], channelsM[i], p_hist);
    p_hist += hist_len; // Skip onto next channel
  }
}

template<typename imType, typename binType>
void IntegralHistogram<imType, binType>::integralHistogramJoint(
  InputArray val, InputArray mag, std::vector<binType>& hist,
  int nmag  
)
/* Joint histogram of val and mag */
{
  CV_Assert( val.channels() == mNChannels );
  CV_Assert( mag.depth() == val.depth() );
  
  int rows = mDim.height;
  int cols = mDim.width;
  int hist_rows = rows + 1;
  int hist_cols = cols + 1;
  int hist_len = hist_rows * hist_cols * mNBins;
  int row_len = hist_cols * mNBins;
  
  // Allocate vector for integral histogram added row and column
  hist.resize(hist_len * mNChannels);
  
  // Get data array
  binType *p_hist = hist.data();
  
  // Zero out the additional row
  for( int x = 0; x < row_len; x++ ){
    p_hist[x] = 0;
  }
  
  // Zero out the additional column
  for( int y = 1; y < hist_rows; y++  ){
    for( int x = 0; x < mNBins; x++ ){
      p_hist[y * row_len + x] = 0;
    }
  }

  // Split each channel
  std::vector<Mat> channelsV, channelsM;
  if( mNChannels != 1 ){
    split(val, channelsV);
    split(mag, channelsM);
  } else {
    channelsV.push_back(val.getMat());
    channelsM.push_back(mag.getMat());
  }
    
  // Calculate integral histogram for each channel
  for( int i = 0; i < mNChannels; i++ ){
    wavefrontScanJoint(channelsV[i], channelsM[i], p_hist, nmag);
    p_hist += hist_len; // Skip onto next channel
  }
}

template<typename imType, typename binType>
template<typename simType>
void IntegralHistogram<imType, binType>::compare(
  std::vector<binType>& h1, std::vector<binType>& h2,
  Size size, OutputArray out,
  typename Compare<simType>::f cmp
)
{
  int out_cols = mDim.width  - size.width  + 1;
  int out_rows = mDim.height - size.height + 1;
  int hist_rows = mDim.height + 1;
  int hist_cols = mDim.width + 1;
  int hist_len = hist_rows * hist_cols * mNBins;
  
  // Get input histogram data
  const binType *h1d = h1.data();
  const binType *h2d = h2.data();
  
  // Allocate matrix for each channel
  std::vector<Mat> channels;
  for( int i = 0; i < mNChannels; i++ ){
    channels.push_back(
      Mat(out_rows, out_cols, DataType<simType>::type)
    );
  }
  
  // Compare each channel individually
  for( int i = 0; i < mNChannels; i++ ){
    // Compare single channel
    compSingle<simType>(h1d, h2d, size, channels[i], cmp);
    
    // Skip onto next channel
    h1d += hist_len;
    h2d += hist_len;
  }
  
  // Merge all channels to single matrix
  Mat merged;
  merge(channels, merged);
  
  // Return the result
  merged.copyTo(out);
}

template<typename imType, typename binType>
void IntegralHistogram<imType, binType>::regionHistogram(
  const std::vector<binType>& integral, const Rect &region,
  std::vector<binType>& out
)
{
  int rows = mDim.height;
  int cols = mDim.width;
  int hist_rows = rows + 1;
  int hist_cols = cols + 1;
  int hist_len = hist_rows * hist_cols * mNBins;
  out.resize(mNBins * mNChannels);
  const binType *idata = integral.data();
  binType *odata = out.data();
  
  for( int ch = 0; ch < mNChannels; ch++ ) {
    int x0 = region.x;
    int x1 = region.x + region.width;
    int y0 = region.y;
    int y1 = region.y + region.height;

    int row_len = (mDim.width + 1) * mNBins;
    const binType *h00 = idata + x0 * mNBins + y0 * row_len;
    const binType *h01 = idata + x1 * mNBins + y0 * row_len;
    const binType *h10 = idata + x0 * mNBins + y1 * row_len;
    const binType *h11 = idata + x1 * mNBins + y1 * row_len;

    regionHist(h00, h01, h10, h11, odata);
    
    // skip to next channel
    idata += hist_len;
    odata += mNBins;
  }
}

template<typename imType, typename binType>
void IntegralHistogram<imType, binType>::wavefrontScan(
  Mat_<imType> image, binType* hist
)
{
  int x, y;
  int hx, hy;
  int h00, h01, h10, h11;
  int bin;
  int row_len = (image.cols + 1)*mNBins;
  
  for( y = 0, hy = 0; y < image.rows; y++, hy += row_len ){
    imType *p_row = image.template ptr<imType>(y);
    for( x = 0, hx = 0; x < image.cols; x++, hx += mNBins ){
      // Calculate histogram coordinates
      h00 = hx +           hy;
      h01 = hx +           hy + row_len;
      h10 = hx + mNBins  + hy;
      h11 = hx + mNBins  + hy + row_len;
         
      // Sum left, upper and upper-left histogram
      sumHist(hist + h00, hist + h01, hist + h10, hist + h11);
      
      // Calculate bin index
      bin = (p_row[x] * (mNBins - 1)) / mMaxVal;
      
      // Add the current pixel's bin
      hist[h11+bin]++;
    }
  }
}

template<typename imType, typename binType>
void IntegralHistogram<imType, binType>::wavefrontScanVM(
  Mat_<imType> val, Mat_<imType> mag, binType* hist
)
/* Wavefront scan using value and magnitude */
{
  int x, y;
  int hx, hy;
  int h00, h01, h10, h11;
  int bin;
  int row_len = (mDim.width + 1)*mNBins;
  
  for( y = 0, hy = 0; y < mDim.height; y++, hy += row_len ){
    imType *p_val = val.template ptr<imType>(y);
    imType *p_mag = mag.template ptr<imType>(y);
    for( x = 0, hx = 0; x < mDim.width; x++, hx += mNBins ){
      // Calculate histogram coordinates
      h00 = hx +           hy;
      h01 = hx +           hy + row_len;
      h10 = hx + mNBins  + hy;
      h11 = hx + mNBins  + hy + row_len;
      
      // Calculate bin index
      bin = (p_val[x] * (mNBins - 1)) / mMaxVal;
         
      // Sum left, upper and upper-left histogram
      sumHist(hist + h00, hist + h01, hist + h10, hist + h11);
      
      // Add the current pixel's bin magnitude
      hist[h11+bin] += p_mag[x];
    }
  }
}

template<typename imType, typename binType>
void IntegralHistogram<imType, binType>::wavefrontScanJoint(
  Mat_<imType> val, Mat_<imType> mag, binType* hist, int nmag
)
/* Wavefront scan using value and magnitude */
{
  int x, y;
  int hx, hy;
  int h00, h01, h10, h11;
  int bin;
  int row_len = (mDim.width + 1)*mNBins;
  int nval = mNBins / nmag;
  
  for( y = 0, hy = 0; y < mDim.height; y++, hy += row_len ){
    imType *p_val = val.template ptr<imType>(y);
    imType *p_mag = mag.template ptr<imType>(y);
    for( x = 0, hx = 0; x < mDim.width; x++, hx += mNBins ){
      // Calculate histogram coordinates
      h00 = hx +           hy;
      h01 = hx +           hy + row_len;
      h10 = hx + mNBins  + hy;
      h11 = hx + mNBins  + hy + row_len;
      
      // Calculate bin index
      bin =  (p_val[x] * (nval - 1)) / mMaxVal;
      bin += ((p_mag[x] * (nmag - 1)) / 
        std::numeric_limits<imType>::max()) * nval;
         
      // Sum left, upper and upper-left histogram
      sumHist(hist + h00, hist + h01, hist + h10, hist + h11);
      
      // Add the current pixel's bin magnitude
      hist[h11+bin]++;
    }
  }
}

template<typename imType, typename binType>
template<typename simType>
void IntegralHistogram<imType, binType>::compSingle(
  const binType *h1, const binType *h2, Size size, Mat_<simType> out,
  typename Compare<simType>::f cmp
)
/* Compares single channel integral histograms */
{
  int out_cols = mDim.width  - size.width  + 1;
  int out_rows = mDim.height - size.height + 1;

  // Allocate memory for output histograms
  std::vector<binType> res1, res2;
  res1.resize(mNBins);
  res2.resize(mNBins);
  binType *res1d = res1.data();
  binType *res2d = res2.data();
  
  // Pre-calculated values for performance boost
  int row_len = (mDim.width + 1) * mNBins;
  int width = size.width * mNBins;
  int height = size.height * row_len;
  
  int h00, h01, h10, h11;
  int x, y;   // Image index
  int hx, hy; // Integral histogram index

  for( y = 0, hy = 0; y < out_rows; y++, hy += row_len )
  {
    simType *p_row = out.template ptr<simType>(y);
    
    for( x = 0, hx = 0; x < out_cols; x++, hx += mNBins ){
      h00 = hx         + hy;
      h01 = hx + width + hy;
      h10 = hx         + hy + height;
      h11 = hx + width + hy + height;
      
      regionHist( h1+h00, h1+h01, h1+h10, h1+h11, res1d);
      regionHist( h2+h00, h2+h01, h2+h10, h2+h11, res2d);
      p_row[x] = cmp(res1d, res2d, mNBins);
    }
  }
}

template<typename imType, typename binType>
inline void IntegralHistogram<imType, binType>::sumHist(
  binType* hist00, binType* hist01, binType* hist10, binType* hist11
)
{
  for(int i = 0; i < mNBins; i++){
    hist11[i] = hist01[i] + hist10[i] - hist00[i];
  }
}

template<typename imType, typename binType>
inline void IntegralHistogram<imType, binType>::regionHist(
  const binType* hist00, const binType* hist01, const binType* hist10,
  const binType* hist11, binType* out
)
{
  for(int i = 0; i < mNBins; i++){
    out[i] = hist11[i] - hist01[i] - hist10[i] + hist00[i];
  }
}
