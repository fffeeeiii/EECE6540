//==============================================================
// EECE.6540 Lab 3
//
// Image Rotate with DPC++
//
// Author: Fei Zhou
//
// Copyright ©  2022-
//
// MIT License
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <cmath>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

// useful header files for image convolution
#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"


using Duration = std::chrono::duration<double>;
class Timer {
 public:
  Timer() : start(std::chrono::steady_clock::now()) {}

  Duration elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start);
  }

 private:
  std::chrono::steady_clock::time_point start;
};

static const char* inputImagePath = "./Images/cat.bmp";

#define Theta 45        // Rotate Angle (degree)
#define PI 3.14159265358979

#define IMAGE_SIZE (720*1080)
constexpr size_t array_size = IMAGE_SIZE;
typedef std::array<float, array_size> FloatArray;

//************************************//
// Image Rotate in DPC++ on device: 
//************************************//
void ImageRotate(queue &q, float *image_in, float *image_out, float sinTheta, 
    float cosTheta, const size_t ImageRows, const size_t ImageCols) 
{

    // We create buffers for the input and output data.
    //
    buffer<float, 1> image_in_buf(image_in, range<1>(ImageRows*ImageCols));
    buffer<float, 1> image_out_buf(image_out, range<1>(ImageRows*ImageCols));

    //for(int i=0; i<ImageRows; i++) {
    //  for(int j=0; j<ImageCols; j++)
    //    std::cout << "image_out[" << i << "," << j << "]=" << (float *)image_out[i*ImageCols+j] << std::endl;  
    //}
 
    // Create the range object for the pixel data.
    range<2> num_items{ImageRows, ImageCols};

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](handler &h) {
      // Create an accessor to buffers with access permission: read, write or
      // read/write. The accessor is a way to access the memory in the buffer.
      accessor srcPtr(image_in_buf, h, read_only);

      // Another way to get access is to call get_access() member function 
      auto dstPtr = image_out_buf.get_access<access::mode::write>(h);

      // Use parallel_for to run image rotate in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // DPC++ supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](id<2> item)  { 

        // get row and col of the pixel assigned to this work item
        int row = item[0];
        int col = item[1];       
        
        // calculate the new position (xpos, ypos) 
        //     from (row, col) around (ImageRows/2, ImageCols/2) 
        int ix = row - (int)ImageRows/2;
        int iy = col - (int)ImageCols/2;     
        //float xpos = ((float)ix)*cosTheta + ((float)iy)*sinTheta;
        //float ypos = -1.0f*((float)ix)*sinTheta + ((float)iy)*cosTheta;
        float xpos = row;
        float ypos = col;
    
        // Bound checking 
        if(((int)xpos >= 0) && ((int)xpos < ImageRows) && ((int)ypos >= 0) && ((int)ypos < ImageCols)) {
        	// Write the new pixel value
        	dstPtr[xpos*ImageCols+ypos] = srcPtr[row*ImageCols+col];
        }
      });

    });

}


int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA || FPGA_PROFILE
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  float *hInputImage;
  float *hOutputImage;

  int imageRows;
  int imageCols;
  int i;

  // Set the rotate angle Theta
  float sinTheta = sin(Theta * PI / 180.0f);
  float cosTheta = cos(Theta * PI / 180.0f);
  printf("Rows and Cols = %d, %d \n", imageRows, imageCols);  

#ifndef FPGA_PROFILE
  // Query about the platform
  unsigned number = 0;
  auto myPlatforms = platform::get_platforms();
  // loop through the platforms to poke into
  for (auto &onePlatform : myPlatforms) {
    std::cout << ++number << " found .." << std::endl << "Platform: " 
    << onePlatform.get_info<info::platform::name>() <<std::endl;
    // loop through the devices
    auto myDevices = onePlatform.get_devices();
    for (auto &oneDevice : myDevices) {
      std::cout << "Device: " 
      << oneDevice.get_info<info::device::name>() <<std::endl;
    }
  }
  std::cout<<std::endl;
#endif

  /* Read in the BMP image */
  hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
  printf("imageRows=%d, imageCols=%d\n", imageRows, imageCols);
  
  /* Allocate space for the output image */
  hOutputImage = (float *)malloc( imageRows*imageCols * sizeof(float) );
  for(i=0; i<imageRows*imageCols; i++) {
    hOutputImage[i] = 1234.0;
  }

  Timer t;

  try {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Image rotate in DPC++
    ImageRotate(q, hInputImage, hOutputImage, sinTheta, cosTheta, imageRows, imageCols);
  } catch (exception const &e) {
    std::cout << "An exception is caught for image convolution.\n";
    std::terminate();
  }

  std::cout << t.elapsed().count() << " seconds\n";

  /* Save the output bmp */
  printf("Output image saved as: cat-rotated.bmp\n");
  writeBmpFloat(hOutputImage, "cat-rotated.bmp", imageRows, imageCols,
          inputImagePath);


//#ifndef FPGA_PROFILE
//  /* Verify result */
//  float *refOutput = convolutionGoldFloat(hInputImage, imageRows, imageCols,
//    filter, filterWidth);
//
//  writeBmpFloat(refOutput, "cat-roated-ref.bmp", imageRows, imageCols,
//          inputImagePath);
//
//  bool passed = true;
//  for (i = 0; i < imageRows*imageCols; i++) {
//    if (fabsf(refOutput[i]-hOutputImage[i]) > 0.001f) {
//        printf("%f %f\n", refOutput[i], hOutputImage[i]);
//        passed = false;
//    }
//  }
//  if (passed) {
//    printf("Passed!\n");
//    std::cout << "Image Rotate successfully completed on device.\n";
//  }
//  else {
//    printf("Failed!\n");
//  }
//#endif


  return 0;
}
