// Skeleton code for B657 A4 Part 3.
// by  D. Crandall
//
// Run like this, for example:
//   ./stereo part3/Aloe/view1.png part3/Aloe/view5.png part3/Aloe/gt.png
// and output files will appear in part3/Aloe
//
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <CImg.h>
#include <assert.h>

using namespace cimg_library;
using namespace std;

double sqr(double a) { return a*a; }

// This code may or may not be helpful. :) It computes a 
//  disparity map by looking for best correspondences for each
//  window independently (no MRF).
//
CImg<double> naive_stereo(const CImg<double> &input1, const CImg<double> &input2, int window_size, int max_disp)
{  
  CImg<double> result(input1.width(), input1.height());

  for(int i=0; i<input1.height(); i++)
    for(int j=0; j<input1.width(); j++)
      {
	pair<int, double> best_disp(0, INFINITY);

	for (int d=0; d < max_disp; d++)
	  {
	    double cost = 0;
	    for(int ii = max(i-window_size, 0); ii <= min(i+window_size, input1.height()-1); ii++)
	      for(int jj = max(j-window_size, 0); jj <= min(j+window_size, input1.width()-1); jj++)
		cost += sqr(input1(min(jj+d, input1.width()-1), ii) - input2(jj, ii));

	    if(cost < best_disp.second)
	      best_disp = make_pair(d, cost);
	  }
	result(j,i) = best_disp.first;
      }

  return result;
}

// implement this!
//  this placeholder just returns a random disparity map
//

double pair_cost(const CImg<double> &temp, const int &x, const int &y,int &l){
  double cost = 4;
  if(x>0 and x<temp.width()-1 and y>0 and y<temp.height()-1){
    if(l == temp(x-1,y,0,0)) cost -= 1;
    if(l == temp(x+1,y,0,0)) cost -= 1;
    if(l == temp(x,y-1,0,0)) cost -= 1;
    if(l == temp(x,y+1,0,0)) cost -= 1;
  }
  //if(cost >= 1) cost = 600;
  //else cost = 0;
  return cost;
}


int find_d(const CImg<double> &input1, const CImg<double> &input2, const int&w, const int&max_d, CImg<double> &temp,int &x, int&y){
  double a = 50;
  double min_cost = INFINITY;
  int best_d = 0;
  for(int d=0; d< max_d; d++){
    double cost = 0;
    double c =0;
    double p_cost = pair_cost(temp,x,y,d)*a;
    for(int ii = max(y-w, 0); ii <= min(y+w, input1.height()-1); ii++)
      for(int jj = max(x-w, 0); jj <= min(x+w, input1.width()-1); jj++){
	cost += sqr(input1(min(jj+d, input1.width()-1), ii) - input2(jj, ii));
	c += 1;
      }
    //cout<<cost<<endl;
    cost = cost / c;
    //cout<<cost<<endl;
    cost += p_cost;
    
    if(cost < min_cost){
      min_cost = cost;
      best_d = d;
    }
  }
  //cout<<best_d<<endl;
  return best_d;
}


CImg<double> mrf_stereo(const CImg<double> &img1, const CImg<double> &img2,int w,int d)
{

  CImg<double> result(img1.width(), img1.height(),1,1,-1);

  for(int k=0; k<15;k++){
    cout<<k<<endl;
  for(int j=0; j<img1.height(); j++)
    for(int i=0; i<img1.width(); i++){
      result(i,j,0,0) = find_d(img1,img2,w,d,result,i,j);
    }
  }
  return result;
}



int main(int argc, char *argv[])
{
  if(argc != 4 && argc != 3)
    {
      cerr << "usage: " << argv[0] << " image_file1 image_file2 [gt_file]" << endl;
      return 1;
    }

  string input_filename1 = argv[1], input_filename2 = argv[2];
  string gt_filename;
  if(argc == 4)
    gt_filename = argv[3];

  // read in images and gt
  CImg<double> image1(input_filename1.c_str());
  CImg<double> image2(input_filename2.c_str());
  CImg<double> gt;

  if(gt_filename != "")
  {
    gt = CImg<double>(gt_filename.c_str());

    // gt maps are scaled by a factor of 3, undo this...
    for(int i=0; i<gt.height(); i++)
      for(int j=0; j<gt.width(); j++)
	gt(j,i) = gt(j,i) / 3.0;
  }
  
  // do naive stereo (matching only, no MRF)
  CImg<double> naive_disp = naive_stereo(image1, image2, 5, 100);
  naive_disp.get_normalize(0,255).save((input_filename1 + "-disp_naive.png").c_str());

  // do stereo using mrf
  CImg<double> mrf_disp = mrf_stereo(image1, image2,5,100);
  mrf_disp.get_normalize(0,255).save((input_filename1 + "-disp_mrf.png").c_str());

  // Measure error with respect to ground truth, if we have it...
  if(gt_filename != "")
    {
      cout << "Naive stereo technique mean error = " << (naive_disp-gt).sqr().sum()/gt.height()/gt.width() << endl;
      cout << "MRF stereo technique mean error = " << (mrf_disp-gt).sqr().sum()/gt.height()/gt.width() << endl;

    }

  return 0;
}
