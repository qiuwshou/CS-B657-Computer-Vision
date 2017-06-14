// Skeleton code for B657 A4 Part 2.
// D. Crandall
//
//
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <CImg.h>
#include <assert.h>
//#include "part2.h"
using namespace cimg_library;
using namespace std;


class Point
{
public:
  Point() {}
  Point(int _col, int _row) : row(_row), col(_col) {}
  int row, col;
};
int ip(const vector<Point> &fg, const vector<Point> &bg, const int &x, const int &y);
CImg<double>  log_cost(const CImg<double> &img, const vector<Point> &fg);

int ip(const vector<Point> &fg, const vector<Point> &bg, const int &x, const int &y){
  
  int p_label = -1;
  for(int i=0;i<fg.size();i++)
    if(x==fg[i].col and y==fg[i].row)
      p_label = 1;  

  for(int i=0;i<bg.size();i++)
    if(x==bg[i].col and y==bg[i].row)
      p_label = 0;

  return p_label;
}

CImg<double>  log_cost(const CImg<double> &img, const vector<Point> &fg){
  double pi = 3.1415926535897;
  CImg<double> mean(1,1,1,3,0);
  CImg<double> variance(1,1,1,3,0);  
  CImg<double> sum(1,1,1,3,0);
  CImg<double> ssd(1,1,1,3,0);
  CImg<double> result(img.width(),img.height(),1,1,0);
  for(int i=0; i<fg.size(); i++)
    for(int k=0; k<3;k++)
      sum(0,0,0,k) = sum(0,0,0,k) + img(fg[i].col,fg[i].row,0,k);

  for(int k=0;k<3;k++){
    mean(0,0,0,k)= sum(0,0,0,k)/fg.size();
    //cout<<mean(0,0,0,k)<<endl;
  }

  for(int i=0; i<fg.size(); i++)
    for(int k=0;k<3;k++)
      ssd(0,0,0,k) = ssd(0,0,0,k) + pow(img(fg[i].col,fg[i].row,0,k)-mean(0,0,0,k),2);

  for(int k=0;k<3;k++){
    variance(0,0,0,k)= ssd(0,0,0,k)/fg.size();
    //cout<<variance(0,0,0,k)<<endl;
  }

  for(int i=0;i<img.width();i++){
    for(int j=0;j<img.height();j++){
      double determinant = variance(0,0,0,0)*variance(0,0,0,1)*variance(0,0,0,2);
      double e=0;
      for(int k=0; k<3;k++){
	e=e+pow((img(i,j,0,k)-mean(0,0,0,k)),2)/variance(0,0,0,k);
	//cout<<e<<endl;
      }
      //cout<<e<endl;
      e = exp(e*-0.5);
      //cout<<e<<endl;
      double gaussian= e/(sqrt(determinant*pow(2*pi,3)));
      //cout<<-log(gaussian)<<endl;
      result(i,j,0,0) = -log(gaussian);
    }
  }
  
  
  return result;
}




CImg<double> naive_segment(const CImg<double> &img, const vector<Point> &fg, const vector<Point> &bg,const CImg<double> &cost)
{
  // implement this in step 2...
  //  this placeholder just returns a random disparity map
  CImg<double> result(img.width(), img.height());
  //CImg<double> cost = log_cost(img,fg);
  //CImg<double> m("mean.png");
  //CImg<double> v("variance.png");
  double b = 18;
  int p;
  for(int i=0; i<img.width(); i++){
    for(int j=0; j<img.height(); j++){
      p = ip(fg,bg,i,j);
      
      if(p==1){
	result(i,j,0,0)=1;
      }
      else if(p==0){
	result(i,j,0,0)=0;
      }
      else if(p==-1){
	if(b <= cost(i,j,0,0)){
	  result(i,j,0,0) = 0;
	}
	else{ result(i,j,0,0) = 1;}
      }
    }
  }

  return result;
}

double pair_cost(CImg<double> &temp,const int &l,const int &x, const int &y){
  double cost = 4;
  if(x>0 and x<temp.width()-1 and y>0 and y<temp.height()-1){
    if(l == temp(x-1,y,0,0)) cost -= 1;
    if(l == temp(x+1,y,0,0)) cost -= 1;
    if(l == temp(x,y-1,0,0)) cost -= 1;
    if(l == temp(x,y+1,0,0)) cost -= 1;
  }
  return cost;
}


CImg<double> mrf_segment(const CImg<double> &img, const vector<Point> &fg, const vector<Point> &bg,CImg<double> &cost)
{
  double a = 1, b=18;
  CImg<double> temp=img;
  //initialize label
  for(int i=0; i<img.width();i++){
    for(int j=0; j<img.height();j++){
      int p = ip(fg,bg,i,j);
      if(p==0 or p==1) 
	temp(i,j,0,0) = p;
      else
	temp(i,j,0,0) = -1;
    }
  } 
  
  //cout<<img.width()<<" "<<img.height()<<endl;
  //cout<<temp.width()<<" "<<temp.height()<<endl;
  //for(int k=0; k<img.width()+img.height();k++){
  for(int k=0; k<25;k++){
    cout<<k<<endl;
  for(int i=0; i<img.width();i++){
    for(int j=0; j<img.height();j++){
      int p=ip(fg,bg,i,j);
      if(p==0 or p==1) temp(i,j,0,0) = p;
      else{
	double p_0 = pair_cost(temp,0,i,j);
	double p_1 = pair_cost(temp,1,i,j);
        //cout<<p_0<<" "<<p_1<<endl;
        double cost_1 = a*p_1 + cost(i,j,0,0);
        double cost_0 = a*p_0 + b;
	if(cost_0 < cost_1){
	  temp(i,j,0,0) = 0;}
	else{
	  temp(i,j,0,0) = 1;
	}
      }
    }
  }
  }
  return temp;
}

// Take in an input image and a binary segmentation map. Use the segmentation map to split the 
//  input image into foreground and background portions, and then save each one as a separate image.
//
// You'll just need to modify this to additionally output a disparity map.
//
void output_segmentation(const CImg<double> &img, const CImg<double> &labels, const string &fname)
{
  // sanity checks. If one of these asserts fails, you've given this function invalid arguments!
  
  assert(img.height() == labels.height());
  assert(img.width() == labels.width());

  CImg<double> img_fg = img, img_bg = img, img_disp = img;

  for(int i=0; i<labels.height(); i++)
    for(int j=0; j<labels.width(); j++)
      {  
	//cout<<"x:"<<j<<"y:"<<i<<labels(j,i)<<endl;
	if(labels(j,i) == 1){
	  img_fg(j,i,0,0) = img_fg(j,i,0,1) = img_fg(j,i,0,2) = 0;
	  img_disp(j,i,0,0) = img_disp(j,i,0,1) = img_disp(j,i,0,2) = 255;
	}
	else if(labels(j,i) == 0){
	  img_bg(j,i,0,0) = img_bg(j,i,0,1) = img_bg(j,i,0,2) = 0;
	  img_disp(j,i,0,0) = img_disp(j,i,0,1) = img_disp(j,i,0,2) = 0;
	}
	else
	  assert(0);
      }

  img_fg.get_normalize(0,255).save((fname + "_fg.png").c_str());
  img_bg.get_normalize(0,255).save((fname + "_bg.png").c_str());
  img_disp.get_normalize(0,255).save((fname + "_disp.png").c_str());
}

int main(int argc, char *argv[])
{
  if(argc != 3)
    {
      cerr << "usage: " << argv[0] << " image_file seeds_file" << endl;
      return 1;
    }

  string input_filename1 = argv[1], input_filename2 = argv[2];

  // read in images and gt
  CImg<double> image_rgb(input_filename1.c_str());
  CImg<double> seeds_rgb(input_filename2.c_str());

  // figure out seed points 
  vector<Point> fg_pixels, bg_pixels;
  for(int i=0; i<seeds_rgb.height(); i++)
    for(int j=0; j<seeds_rgb.width(); j++)
      {
	// blue --> foreground
	if(max(seeds_rgb(j, i, 0, 0), seeds_rgb(j, i, 0, 1)) < 100 && seeds_rgb(j, i, 0, 2) > 100)
	  fg_pixels.push_back(Point(j, i));

	// red --> background
	if(max(seeds_rgb(j, i, 0, 2), seeds_rgb(j, i, 0, 1)) < 100 && seeds_rgb(j, i, 0, 0) > 100)
	  bg_pixels.push_back(Point(j, i));
      }

  //calculate mean and variance of front points
  //CImg<double> log_cost = log(image_rgb,fg_pixels);

  // do naive segmentation
  CImg<double> log_c = log_cost(image_rgb,fg_pixels);
  CImg<double> labels_naive = naive_segment(image_rgb, fg_pixels, bg_pixels,log_c);
  output_segmentation(image_rgb, labels_naive, input_filename1 + "-naive_segment_result");

  // do mrf segmentation
  CImg<double>labels = mrf_segment(image_rgb, fg_pixels, bg_pixels,log_c);
  output_segmentation(image_rgb, labels, input_filename1 + "-mrf_segment_result");

  return 0;
}
