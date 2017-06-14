// Skeleton code for B656 A4 Part 1.
// D. Crandall
//
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <CImg.h>
#include <assert.h>
#include <string>
using namespace cimg_library;
using namespace std;

int main(int argc, char *argv[])
{
  if(argc != 3)
    {
      cerr << "usage: " << argv[0] << " image_file disp_file" << endl;
      return 1;
    }

  string input_filename1 = argv[1], input_filename2 = argv[2];
  vector<double> factor;
  for(double i=0; i < 0.1; i=i+0.01)
    factor.push_back(i);
 
  // read in images and gt
  CImg<double> image_rgb(input_filename1.c_str());
  CImg<double> image_disp(input_filename2.c_str());     
  CImg<double> image_result = image_rgb;

  double f;
  for(int c=0; c<factor.size();c++){
    f=factor[c];
  for(int i=0; i<image_rgb.width(); i++){
    for(int j=0; j<image_rgb.height(); j++){
      int d = (int)image_disp(i,j,0,0) * f;
      if( i+d <= image_rgb.width()){
	image_result(i,j,0,1) = image_rgb(i+d,j,0,1);
	image_result(i,j,0,2) = image_rgb(i+d,j,0,2);
      }
    }
  }  

  image_result.get_normalize(0,255).save((input_filename1 + "-stereogram.png_"+to_string(f)).c_str());
  }
  return 0;
}
