// B657 assignment 2 skeleton code
//
// Compile with: "make"
//
// See assignment handout for command line and project specifications.


//Link to the header file
#include "CImg.h"
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <Sift.h>
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>  
//Use the cimg namespace to access the functions easily
using namespace cimg_library;
using namespace std;

class ImagePoint
{
	public:
//	ImagePoint(int col,int row) : x(col), y(row) {};
	int x,y;
};

class SiftPointPair
{
	public:
	ImagePoint point, point1, point2;
	double distance1,distance2;
};


class MatchCount
{
	public:
	CImg<double> image;
	int match_count;
	string file_name;
};

typedef struct{
  int d1, d2;
  int v_x, v_y;
  int x1, y1, x2, y2;
}sift_pairs;

double dist(vector<float> &d1, vector<float> &d2){
  double sum = 0;
  for(int i =0; i < 128; i++){
    sum += pow(( d1[i] - d2[i]),2) ;
  }
  sum = sqrt(sum);
  return sum;
}


void compare_des(vector<SiftDescriptor> &v1, vector<SiftDescriptor> &v2, float thresh, vector< sift_pairs> &pairs){
  float first , second;
  int first_id , second_id;
  float d;
  sift_pairs pair;



  for(int i = 0; i < v1.size(); i++){
    first = 99999;
    second = 99999;
    first_id = -1;
    second_id = -1;
    for(int j = 0; j < v2.size(); j++){
      d = dist(v1[i].descriptor, v2[j].descriptor);
      if( d < first){
	second = first;
	second_id = first_id;
	first = (double)d;
	first_id = j;
      }
      if(first_id != -1 && second_id != -1){
	d = first / second;
	//cout<<"d: "<<d<<endl;
	if( d < thresh){
	  //do somethin
	  pair.d1 = i; pair.d2 = first_id;
          pair.x1 = v1[i].col; pair.y1 = v1[i].row;
	  pair.x2 = v2[first_id].col ; pair.y2 = v2[first_id].row;
	  pair.v_x = v2[first_id].col - v1[i].col;
	  pair.v_y = v2[first_id].row - v1[i].row;
	  //cout<< i <<" "<<first_id << " "<<pair.v_x<< " " << pair.v_y<<endl;
	  pairs.push_back(pair);
	}
      }
    } 
  }
}

void draw(CImg<double> &input, vector<SiftDescriptor> &v1, CImg<double> &test, vector<SiftDescriptor> &v2, vector< sift_pairs> &pairs){
  int w = input.width() + test.width();
  int h = max(input.height(), test.height());
  CImg<double> output(w, h, 1, 3);
  output.fill(0);
  /*  for(int i = 0; i < w; i++){
    for(int j = 0; j < h; j++){
      output(i, j, 0, 0) = 0;
      output(i, j, 0, 1) = 0;
      output(i, j, 0, 2) = 0;
    }
    }*/

  for(int i = 0; i < input.width(); i++){
    for(int j = 0; j < input.height(); j++){
      output(i, j, 0, 0) = input(i ,j, 0, 0);
      output(i,j, 0, 1) = input(i ,j, 0, 1);
      output(i,j, 0, 2) = input(i ,j, 0, 2);
    }
  }
  for(int i = 0; i < test.width(); i++){
    for(int j = 0; j < test.height(); j++){
      output(i+input.width(), j, 0, 0) = test(i, j , 0, 0);
      output(i+input.width(), j, 0, 1) = test(i, j , 0, 1);
      output(i+input.width(), j, 0, 2) = test(i, j , 0, 2);
    }
  }

  int d1, d2, x0, y0, x1, y1;
  const double color[] = {255, 0, 0};
  CImgList<int> points;
  for(int i = 0; i < pairs.size(); i++){
    d1 = pairs[i].d1;
    d2 = pairs[i].d2;
    x0 = v1[d1].col;
    y0 = v1[d1].row;
    x1 = v2[d2].col;
    y1 = v2[d2].row;
    //cout<<"i "<<i<<":"<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<endl;
    output.draw_line(x0, y0, x1+input.width(), y1,  color);
  }
 output.get_normalize(0,255).save("draw_line.png");

}

//part 2.2
CImg<double>  proj_transform(vector< sift_pairs> &record){
  //calculate inverse proj_matrix, move the test image onto input image
  CImg<double> b(1,8,1,1);
  CImg<double> a(8,8,1,1);
  CImg<double> output(3,3,1,1);
  int x, y, x_prime, y_prime, w, h1, h2;
  for(int i=0; i < record.size(); i++){
    x_prime = record[i].x2; y_prime = record[i].y2;
    x= record[i].x1; y = record[i].y1;
    cout<<"x: "<<x<<"y: "<<y<<"x': "<<x_prime<<"y': "<<y_prime<<endl;
    h1 = 2*i; h2 = 2*i+1;
    b(0,h1,0,0) = x_prime; b(0,h2,0,0) = y_prime;
    for(int j=3; j <=5 ; j++)
      a(j,h1,0,0) = 0;
    for(int j=0; j <=2; j++)
      a(j,h2,0,0) = 0;
    a(0,h1,0,0) = x; a(1,h1,0,0) = y; a(2,h1,0,0) = 1; a(6,h1,0,0)= - x*x_prime;a(7,h1,0,0) = -y*x_prime;
    a(3,h2,0,0) = x; a(4,h2,0,0) = y; a(5,h2,0,0) = 1; a(6,h2,0,0)= - x*y_prime;a(7,h2,0,0) = -y*y_prime; 
  }
  b.solve(a);
  /*for(int i=0; i <8; i++){
    cout<<b(0,i,0,0) <<endl;
    }*/
  for(int i=0; i < 3;i++){
    for(int j=0; j <3; j++){
      if(j==2 && i==2){
	output(j,i,0,0) = 1;
      }
      else{
        output(j,i,0,0) = b(0,i*3+j,0,0);
      }
    }
  }
  cout<<"{ "<< output(0,0,0,0)<<", "<<output(1,0,0,0)<<", "<<output(2,0,0,0)<<" }"<<endl;
  cout<<"{ "<< output(0,1,0,0)<<", "<<output(1,1,0,0)<<", "<<output(2,1,0,0)<<" }"<<endl;
  cout<<"{ "<< output(0,2,0,0)<<", "<<output(1,2,0,0)<<", "<<output(2,2,0,0)<<" }"<<endl; 
  return output;
}
void warp_image(CImg<double> &output, CImg<double>blend, CImg<double> &test, CImg<double> &matrix, string &s){

  double x_prime, y_prime, w_prime;
  int x2, y2;
  
  for(int x=0; x < output.width(); x++){
    for(int y=0; y < output.height(); y++){
      x_prime = matrix(0,0,0,0)*x + matrix(1,0,0,0)*y + matrix(2,0,0,0);
      y_prime = matrix(0,1,0,0)*x + matrix(1,1,0,0)*y + matrix(2,1,0,0);
      w_prime = matrix(0,2,0,0)*x + matrix(1,2,0,0)*y + matrix(2,2,0,0);
      x2 = x_prime / w_prime;
      y2 = y_prime / w_prime;
      //cout<<"x2: "<<x2<<"y2: "<<y2<<endl;
      if( x2 >= 0 && x2 < test.width() && y2 >= 0 && y2 < test.height()){
	blend(x,y,0,0) += test(x2,y2,0,0)*0.5+0.5*blend(x,y,0,0);
	blend(x,y,0,1) += test(x2,y2,0,1)*0.5+0.5*blend(x,y,0,1);;
	blend(x,y,0,2) += test(x2,y2,0,2)*0.5+0.5*blend(x,y,0,2);;
	output(x,y,0,0) = test(x2,y2,0,0);
        output(x,y,0,1) = test(x2,y2,0,1);
        output(x,y,0,2) = test(x2,y2,0,2);
      }
    }
  }
  string s1 = s+"_warped.png";
  string s2 = s+"_blend.png";
  blend.get_normalize(0,255).save(s2.c_str());
  output.save(s1.c_str());

}

vector< sift_pairs> RANSAC(/*CImg<double> &input, vector<SiftDescriptor> &v1, CImg<double> &test, vector<SiftDescriptor> &v2,*/ vector< sift_pairs> &pairs, int thresh_x, int thresh_y, int iteration){
  vector< sift_pairs> record;
  int max = -1;
  for(int k = 0; k < iteration; k++){
    vector< sift_pairs> temp;
    int count = 0;
 
  for(int n = 0; n < 4; n++){
    int sample = rand() % pairs.size();
    //cout<<"sample_id: "<< sample<<endl;
    //sift_pairs temp =  pairs[sample];
    temp.push_back(pairs[sample]);
  }

    //for(int i =0; i < temp.size(); i++){
    //int max = -1;
    //int count = 0;
  for(int i = 0; i < 4; i++){ 
    for(int j = 0; j < pairs.size(); j++){
      int dist_x = abs(pairs[j].v_x - temp[i].v_x);
      int dist_y = abs(pairs[j].v_y - temp[j].v_y);
      if(dist_x <= thresh_x && dist_y <= thresh_y){
        count += 1;
      }
    }
  }
  if(count > max){ 
      record = temp;
      max = count;
  }
  }
  
  return record;

}


CImg<double> combine_image(CImg<double> image1, CImg<double> image2, vector<SiftPointPair> pairs, int &match_count)
{
	float threshold = 0.645;
	CImg<double> final_image = image1.get_append(image2, 'x');
	const unsigned char color[] = { 255,215,0 };
	for(int i = 0; i < pairs.size(); i++)
	{
		//cout << "distance1/distance2:" << pairs[i].distance1 <<"/"<< pairs[i].distance2 << endl;
		double ratio = pairs[i].distance1 / pairs[i].distance2;
		//cout << "Ratio : "<< ratio << endl;
		if(ratio < threshold)
		{
			cout << "drawing line" << endl;
			if(pairs[i].distance1 < pairs[i].distance2)
			{
				final_image.draw_line(pairs[i].point.x, pairs[i].point.y, image1.width() + pairs[i].point1.x, pairs[i].point1.y, color);
				for(int j=-2; j<3; j++)
                for(int k=-2; k<3; k++)
                    if(j==0 || k==0)
                        for(int p=0; p<3; p++)
                        {
							if(pairs[i].point.x+k < final_image.width() && pairs[i].point.x+k > 0 && pairs[i].point.y+j < final_image.height() && pairs[i].point.y+j > 0)
							
                            final_image(pairs[i].point.x+k, pairs[i].point.y + j, 0, p) = 0;
							if(pairs[i].point1.x+k < final_image.width() && pairs[i].point1.x+k > 0 && pairs[i].point1.y+j < final_image.height() && pairs[i].point1.y+j > 0)
                            final_image(image1.width() + pairs[i].point1.x+k, pairs[i].point1.y + j, 0, p) = 0;
                            //final_image(pairs[i].point2.x+k, pairs[i].point2.y + j, 0, p) = 0;
                        }

			}
			else
			{
				final_image.draw_line(pairs[i].point.x, pairs[i].point.y, image1.width() + pairs[i].point2.x, pairs[i].point2.y, color);
				for(int j=-2; j<3; j++)
                for(int k=-2; k<3; k++)
                    if(j==0 || k==0)
                        for(int p=0; p<3; p++)
                        {
							if(pairs[i].point.x+k < final_image.width() && pairs[i].point.x+k > 0 && pairs[i].point.y+j < final_image.height() && pairs[i].point.y+j > 0)
                            final_image(pairs[i].point.x+k, pairs[i].point.y + j, 0, p) = 0;
                            //final_image(pairs[i].point1.x+k, pairs[i].point1.y + j, 0, p) = 0;
                            if(pairs[i].point2.x+k < final_image.width() && pairs[i].point2.x+k > 0 && pairs[i].point2.y+j < final_image.height() && pairs[i].point2.y+j > 0)
                            final_image(image1.width() + pairs[i].point2.x+k, pairs[i].point2.y + j, 0, p) = 0;
                        }
			}	
			match_count++;
		}
	}
	cout << "match count:" << match_count;	
	return final_image;
}
double find_euclidean(vector<float> &descriptor1, vector<float> &descriptor2)
{
	double sum = 0.0;
	for(int i=0; i<128; i++)
	{
		double diff = descriptor1[i] - descriptor2[i];
		sum += diff * diff;
	}
	return sqrt(sum);
}

vector<SiftPointPair> find_sift_pairs(vector<SiftDescriptor> descriptors_query, vector<SiftDescriptor> descriptors_input)
{
	double minimum2 = 9999999;
	double minimum1 = 9999999;	
	int minj1 = 0, minj2 = 0;
	vector<SiftPointPair> pairs;
	for(int i=0; i<descriptors_query.size(); i++)
	{
		SiftPointPair pair;
		for(int j=0; j<descriptors_input.size(); j++)
		{
			double distance = find_euclidean(descriptors_query[i].descriptor, descriptors_input[j].descriptor);
			if(distance < minimum1)
			{
				minimum2 = minimum1;
				minimum1 = distance;
				minj2 = minj1;
				minj1 = j;
			}
			//cout << "euclidean:["<<i<<"]["<<j<<"]:"<< distance << endl;
		}
		pair.point.x = descriptors_query[i].col;
		pair.point.y =  descriptors_query[i].row;
		pair.point1.x = descriptors_input[minj1].col;
		pair.point1.y = descriptors_input[minj1].row;
		pair.point2.x = descriptors_input[minj2].col;
		pair.point2.y = descriptors_input[minj2].row;
		pair.distance1 = minimum1;
		pair.distance2 = minimum2;	
		minimum1 = minimum2 = 9999999;
		pairs.push_back(pair);
	}
//	cout << "size of pairs:Happy b'day to me!!!:: "<< pairs.size() << endl;
	return pairs;
}
void warp(CImg<double> &image, string file_name)
{
	cout << "in warp" << endl;
	CImg<double> output_image(image.width(), image.height(), 1, 3, 0);
	double inverted[3][3];
	inverted[0][0] = 1.1246685;
	inverted[0][1] = -0.3146766039759;
	inverted[0][2] = 222.940924688;
	inverted[1][0] = 0.1088390;
	inverted[1][1] = 0.68505866;
	inverted[1][2] = -19.9246953;
	inverted[2][0] = 0.000264587;
	inverted[2][1] = -0.000597068;
	inverted[2][2] = 1.0827848752; 
	double outputpart[3][1];
	double imagepart[3][1];
	for(int row_offset = 0; row_offset < image.height(); row_offset++)
	{
		for(int col_offset = 0; col_offset < image.width(); col_offset++)
		{
//			cout << "["<<row_offset << "][" << col_offset << "]" <<endl;
			
				imagepart[0][0] = col_offset;
				imagepart[1][0] = row_offset;
				imagepart[2][0] = 1;
						
				for(int i = 0; i < 3; i++)
				{
					for(int j = 0; j < 1; j++)
					{
						double sum = 0.0;
						for(int k = 0; k < 3; k++)
						{
							sum += inverted[i][k] * imagepart[k][j];
						}
						outputpart[i][j] = sum;
					}
				}

				outputpart[0][0] = outputpart[0][0] / outputpart[2][0];
				outputpart[1][0] = outputpart[1][0] / outputpart[2][0];

				if(outputpart[0][0] > 0 && outputpart[1][0] > 0 && outputpart[0][0] < image.width() && outputpart[1][0] < image.height())
				{
	        		output_image(col_offset, row_offset, 0) = image(outputpart[0][0], outputpart[1][0], 0);
	        		output_image(col_offset, row_offset, 1) = image(outputpart[0][0], outputpart[1][0], 1);
	        		output_image(col_offset, row_offset, 2) = image(outputpart[0][0], outputpart[1][0], 2);
				}
		}
	}
	
	output_image.get_normalize(0,255).save((file_name + "-warped.png").c_str());
}
bool sort_by_count(MatchCount i, MatchCount j){return (i.match_count > j.match_count);}

int compare_images(CImg<double> input_image1, CImg<double> input_image2)
{

	// convert images to grayscale
	CImg<double> gray1 = input_image1.get_RGBtoHSI().get_channel(2);
	vector<SiftDescriptor> descriptors1 = Sift::compute_sift(gray1);

	CImg<double> gray2 = input_image2.get_RGBtoHSI().get_channel(2);
	vector<SiftDescriptor> descriptors2 = Sift::compute_sift(gray2);

	const double DISTANCE_THRESH = 240;
	const double RATIO_THRESH = 0.9;

	int I = descriptors1.size();
	int J = descriptors2.size();
	double *ssd[I];
	double *euclid_dist[I];
	for(int i=0; i<I; i++)
	{
		ssd[i] = new double[J];
		euclid_dist[i] = new double[J];
	}
	double closest_match[I];
	double second_closest_match[I];
	double match_ratio[I];

	int closest_match_image2[I];

//find the Euclidian distance between each keypoint in image1 vs each keypoint in image2
	for (int i=0; i<I; i++)
	{
	  	for (int j=0; j<J; j++)
	    {
			ssd[i][j]=0;
		 	for (int l=0; l<128; l++)
	  	  	{
	  	      	ssd[i][j] = ssd[i][j] + (descriptors1[i].descriptor[l] - descriptors2[j].descriptor[l])*(descriptors1[i].descriptor[l] - descriptors2[j].descriptor[l]);
		  	}

	  	 	euclid_dist[i][j] = sqrt(ssd[i][j]);
		// cout << "euclid_dist[" << i << "][" << j << "]: " << euclid_dist[i][j] << endl;
	    }
	}
//find closest and 2nd closest matches in image2 to each keypoint in image1
	for (int i=0; i<I; i++)
	{
		closest_match[i] = 10000; second_closest_match[i] = 20000; match_ratio[i] = 0;
	   	for (int j=0; j<J; j++)
	  	{
	  		if ((euclid_dist[i][j] < closest_match[i]) && (euclid_dist[i][j] < DISTANCE_THRESH))
	  	    {	
				second_closest_match[i] = closest_match[i];	// |f1 - f2'|
	  	  		closest_match[i] = euclid_dist[i][j];		// |f1 - f2|
				closest_match_image2[i] = j;
		    
				cout << "euclid_dist[" << i << "][" << j << "]: " << euclid_dist[i][j] << endl;
		    }
		  	else if ((euclid_dist[i][j] < second_closest_match[i]) && (euclid_dist[i][j] < DISTANCE_THRESH))
		    {	
				second_closest_match[i] = euclid_dist[i][j];	 
	  	    }
	    //cout << "closest_match[" << i << "]: " << closest_match[i] << endl;
	    //cout << "second_closest_match[" << i << "]: " << second_closest_match[i] << endl;
	    }
	    match_ratio[i] = closest_match[i]/second_closest_match[i];
	    cout << "match ratio[" << i << "]: " << match_ratio[i] << endl;
	}

	for (int i=0; i<I; i++)
  	{
		delete ssd[i];
   		delete euclid_dist[i];
  	}
 
	CImg<double> merged_images = input_image1;
	CImg<double> annotated_image;

	const unsigned char color[] = { 255 };
	int input_image1_width = input_image1.width();

	merged_images =  merged_images.append(input_image2,'x', 0);

//cout << "input_image1 width: " << input_image1.width() << endl;
//cout << "input_image2 width: " << input_image2.width() << endl;
//cout << "merged_images width: " << merged_images.width() << endl;

//set threshold to select keypoints for plotting
	for (int i=0; i<I; i++)
	{
		if (match_ratio[i] > RATIO_THRESH)
	    { 
 			cout << "match ratio[" << i << "] > " << RATIO_THRESH << ": " << match_ratio[i] << endl;
			cout << "Image1 Desc#" << i << ": x=" << descriptors1[i].col << " y=" << descriptors1[i].row << endl;
	        cout << "Image2 Desc#" << i << ": x=" << descriptors2[closest_match_image2[i]].col << " y=" << descriptors2[closest_match_image2[i]].row << endl;
		merged_images.draw_line(descriptors1[i].col, descriptors1[i].row, descriptors2[closest_match_image2[i]].col + input_image1_width, descriptors2[closest_match_image2[i]].row, color);
	   	}
	}

	annotated_image = merged_images;
	annotated_image.get_normalize(0,255).save("sift_annotated.png");
	
	return 0;
}

int projection_compare(vector<SiftDescriptor> descriptors1, vector<SiftDescriptor> descriptors2, CImg<double> input_image1, CImg<double> input_image2, string output_file)
{
// convert images to grayscale
//	CImg<double> gray1 = input_image1.get_RGBtoHSI().get_channel(2);
//	vector<SiftDescriptor> descriptors1 = Sift::compute_sift(gray1);

//	CImg<double> gray2 = input_image2.get_RGBtoHSI().get_channel(2);
//	vector<SiftDescriptor> descriptors2 = Sift::compute_sift(gray2);

  	int I = descriptors1.size();
  	int J = descriptors2.size();
//	cout << "I = " << I << endl;
//	cout << "J = " << J << endl;
  	int K = 4; 
//    cout << "Enter K [default = " << K << "]: ";
    //cin >> K;
  	double W = 150;
//    cout << "Enter W [default = " << W << "]: ";
    //cin >> W;
  	double RATIO_THRESH = 0.8;
//    cout << "Enter RATIO_THRESH [default = " << RATIO_THRESH << "]: ";
    //cin >> RATIO_THRESH;
  	double DISTANCE_THRESH = 150;
//    cout << "Enter DISTANCE_THRESH [default = " << DISTANCE_THRESH << "]: ";
    //cin >> DISTANCE_THRESH;
  	const int M = 10000;
  	const double N = M;
  	double sum_dot_prod;
  	double *x[K];
  	for (int i=0; i<K; i++)
    {
		x[i] = new double[128];
    }
  	int n, p;

  	double *f1[I];
  	double *f2[J];

  	for(int i=0; i<I; i++)
    {
		f1[i] = new double[K];
    }
  	for(int i=0; i<J; i++)
    {
		f2[i] = new double[K];
    }

  	vector<int> f1_vector[I];
  	vector<int> f2_vector[I];

//generate x_i vector
  	for (int i=0; i<K; i++)
    	for (int j=0; j<128; j++)
      	{
			x[i][j] = (rand()%M)/N;
//       cout<<"x["<<i<<"][" << j << "]="<<x[i][j]<<endl;
      	}

//calculate f1_i(v)
  	for (int a=0; a<I; a++)
    {
		for (int b=0; b<K; b++)
       	{
			sum_dot_prod=0;
			for (int i=0; i<128; i++)
          	{
       	    	sum_dot_prod = sum_dot_prod + x[b][i]*descriptors1[a].descriptor[i];
          	}
        	f1[a][b] = floor(sum_dot_prod/W);
        //cout << "f1[" << a << "][" << b << "]:" << f1[a][b] << endl;
       	}
    }

//calculate f2_i(v)
  	for (int a=0; a<J; a++)
    {
		for (int b=0; b<K; b++)
       	{
			sum_dot_prod=0;
			for (int i=0; i<128; i++)
         	{
       	   		sum_dot_prod = sum_dot_prod + x[b][i]*descriptors2[a].descriptor[i];
         	}
       		f2[a][b] = floor(sum_dot_prod/W);
       //cout << "f2[" << a << "][" << b << "]:" << f2[a][b] << endl;
       	}
    }

//compare f1_i(v) with f2_i(v) to find nearest neighbor pairs
  	for (int a=0; a<I; a++)
    {
		p=0;
     	for (int b=0; b<J; b++)
      	{
			n=0;
       		for (int i=0; i<K; i++)
       	 	{
				if (f1[a][i] == f2[b][i])
	   			{ 
					n=n+1;
	     			if (n == K)
	       			{
					//store matched pairings
						f1_vector[a].push_back(a);
						f2_vector[a].push_back(b);
		
//		cout << "n=" << n << "; f1[" << a << "]= f2[" << b << "]" << endl;
//	        cout << "f1_vector: " << f1_vector[a][p] << endl;
//		cout << "f2_vector: " << f2_vector[a][p] << endl;
						p=p+1;
	       			}
	   			}
         	}
      	}
    }


//calculate Euclidean distance between selected matches
  	int closest_match_vector_element[I];
  	double *ssd[I];
  	double *euclid_dist[I];
  	for (int i=0; i<I; i++)
    {
		ssd[i] = new double[J];
		euclid_dist[i] = new double[J];
    }
  	double closest_match[I];
  	double second_closest_match[I];
  	double match_ratio[I];


  	for (int a=0; a<I; a++)
    {
		for (int p=0; p<f1_vector[a].size(); p++)  
	  	{
			ssd[a][p] = 0;
	   		for (int l=0; l<128; l++)
      	    {
				ssd[a][p] = ssd[a][p] + pow((descriptors1[f1_vector[a][p]].descriptor[l] - descriptors2[f2_vector[a][p]].descriptor[l]), 2);
      	    }
     	   	euclid_dist[a][p] = sqrt(ssd[a][p]);

//     	   cout << "euclid_dist[" << a << "][" << p << "]=" << euclid_dist[a][p] << endl; 
     	}
   	}

//find closest and 2nd closest matches in image2 to each keypoint in image1
  	for (int a=0; a<I; a++)
    {
		closest_match[a] = 10000; second_closest_match[a] = 200000; match_ratio[a] = 0;
     	for (int p=0; p<f1_vector[a].size(); p++)
       	{
     		if ((euclid_dist[a][p] < closest_match[a]) & (euclid_dist[a][p] < DISTANCE_THRESH))
	 		{	
	   			second_closest_match[a] = closest_match[a];	// |f1 - f2'|
	   			closest_match[a] = euclid_dist[a][p];	// |f1 - f2|
	   			closest_match_vector_element[a] = p;

//	   			cout << "euclid_dist[" << a << "][" << p << "]=" << euclid_dist[a][p] << endl;
	 		}

			else if ((euclid_dist[a][p] < second_closest_match[a]) & (euclid_dist[a][p] < DISTANCE_THRESH))
	 		{	
	   			second_closest_match[a] = euclid_dist[a][p];	 
	 		}
       	}

     	match_ratio[a] = closest_match[a]/second_closest_match[a];	  

     //cout << "closest_match[" << a << "]=" << closest_match[a] << endl;
     //cout << "second_closest_match[" << a << "]=" << second_closest_match[a] << endl;
     //cout << "match_ratio[" << a << "]=" << match_ratio[a] << endl;
    }

	CImg<double> merged_images = input_image1;
	CImg<double> annotated_image;

	const unsigned char color[] = { 255 };
	int input_image1_width = input_image1.width();

	merged_images =  merged_images.append(input_image2, 'x', 0);
	int matches = 0;
	//set threshold to select keypoints for plotting
  	for (int a=0; a<I; a++)
    {
		if ((closest_match[a] < DISTANCE_THRESH) | (match_ratio[a] > RATIO_THRESH))
      	{ 
 //			cout << "matches with (distance < DISTANCE_THRES) or (ratio > RATIO_THRESH)" <<endl;
//			cout << "closest_match[" << a << "] > " << DISTANCE_THRESH << ": " << closest_match[a] << endl;
//			cout << "match ratio[" << a << "] > " << RATIO_THRESH << ": " << match_ratio[a] << endl;
      		matches++;
			merged_images.draw_line(descriptors1[f1_vector[a][closest_match_vector_element[a]]].col, descriptors1[f1_vector[a][closest_match_vector_element[a]]].row, descriptors2[f2_vector[a][closest_match_vector_element[a]]].col + input_image1_width, descriptors2[f2_vector[a][closest_match_vector_element[a]]].row, color);
      	}
    }
	   

  	annotated_image = merged_images;
  	annotated_image.get_normalize(0,255).save(output_file.c_str());

	for (int i=0; i<K; i++)
  	delete x[i];

	for (int i=0; i<I; i++)
  	{
		delete f1[i]; delete ssd[i]; delete euclid_dist[i];
	}
	for (int i=0; i<J; i++)
  		delete f2[i];

  	return matches;
}
int main(int argc, char **argv)
{
  	try {

    	if(argc < 2)
      	{
			cout << "Insufficent number of arguments; correct usage:" << endl;
			cout << "    a2-p1 part_id ..." << endl;
			return -1;
      	}

		string query = argv[2];
		vector<string> input_files;
		string part = argv[1];

    	if(part == "part1fast")
      	{
			
			for(int i = 3; i < argc; i++)
			{
				input_files.push_back(argv[i]);
			}
		
			DIR *dir;
			struct dirent *ent;
			string directory(argv[3]);
			if((dir = opendir(directory.c_str())) != NULL)
			{
				input_files.pop_back();
				/* print all the files and directories within directory */
  				while ((ent = readdir (dir)) != NULL) {
					string directory(argv[3]);
					string file(ent->d_name);
					if(ent->d_type == DT_REG)
    				input_files.push_back((file));
  				}
  				closedir (dir);
			}
	// This is just a bit of sample code to get you started, to
	// show how to use the SIFT library.

			CImg<double> input_image1(query.c_str());

	// convert image to grayscale
			CImg<double> gray = input_image1.get_RGBtoHSI().get_channel(2);
			vector<SiftDescriptor> descriptors1 = Sift::compute_sift(gray);

//	CImg<double> input_image1(inputFile1.c_str());
			//CImg<double> input_image2(input_files[0].c_str());

//	compare_images(input_image1, input_image2);
			time_t start;
			time(&start);
			vector<MatchCount> matches;
			for(int image_num = 0; image_num < input_files.size(); image_num++)
            {
				CImg<double> input_image2((directory + input_files[image_num]).c_str());
				CImg<double> gray2 = input_image2.get_RGBtoHSI().get_channel(2);
            	vector<SiftDescriptor> descriptors2 = Sift::compute_sift(gray2);
				MatchCount m;
				string output_file =  input_files[image_num] + "_sift_projection.png";
				m.match_count = projection_compare(descriptors1, descriptors2, input_image1, input_image2, output_file);
                m.file_name = input_files[image_num];
                matches.push_back(m);
			}
			time_t end;
			time(&end);
			double seconds = difftime(end, start);
			cout << "Time Elapsed:" << seconds << " seconds" << endl;
			sort(matches.begin(), matches.end(), sort_by_count);

            cout << "Displaying top 10 results:" << endl;
            for(int i = 0; i < matches.size() && i < 11; i++)
            {
                cout << i<<") matches: " << matches[i].match_count << " file:" << matches[i].file_name << endl;
            }
		}
    	else if(part == "part1")
      	{
			
			for(int i = 3; i < argc; i++)
			{
				input_files.push_back(argv[i]);
			}
		
			DIR *dir;
			struct dirent *ent;
			string directory(argv[3]);
			if((dir = opendir(directory.c_str())) != NULL)
			{
				input_files.pop_back();
				/* print all the files and directories within directory */
  				while ((ent = readdir (dir)) != NULL) {
					string file(ent->d_name);
					if(ent->d_type == DT_REG)
    				input_files.push_back(file);
  				}
  				closedir (dir);
			}
		
		// This is just a bit of sample code to get you started, to
		// show how to use the SIFT library.
			CImg<double> query_image(query.c_str());
			// convert image to grayscale
			CImg<double> gray = query_image.get_RGBtoHSI().get_channel(2);
			vector<SiftDescriptor> descriptors_query = Sift::compute_sift(gray);

//			input_image.get_normalize(0,255).save("sift.png");
			time_t start;
			time(&start);
			vector<MatchCount> matches;
			for(int image_num = 0; image_num < input_files.size(); image_num++)
			{
				CImg<double> *input_image = new CImg<double>((directory + input_files[image_num]).c_str());
				gray = input_image->get_RGBtoHSI().get_channel(2);
				vector<SiftDescriptor> descriptors_input = Sift::compute_sift(gray);
				
				vector<SiftPointPair> pairs = find_sift_pairs(descriptors_query, descriptors_input);
				int mtch_cnt = 0;
				CImg<double> final_image = combine_image(query_image, *input_image, pairs, mtch_cnt);
				string output_file = input_files[image_num] + "_combined_sift.png";	
				final_image.get_normalize(0,255).save((output_file).c_str());
	
				MatchCount m;
				m.image = final_image;
				m.match_count = mtch_cnt;
				m.file_name = input_files[image_num];
				matches.push_back(m);
			}
			time_t end;
			time(&end);

			double seconds = difftime(end, start);
			cout << "Time Elapsed:" << seconds << " seconds" << endl;
			cout << "matches size:" << matches.size();
			sort(matches.begin(), matches.end(), sort_by_count);

			cout << "Displaying top 10 results:" << endl;
			for(int i = 0; i < matches.size() && i < 11; i++)
			{
				cout << "matches: " << matches[i].match_count << " file:" << matches[i].file_name << endl;
			}
	   	}
    	else if(part == "part2.1")
      	{
			// do something here!
			// Part 2.1
			CImg<double> query_image(query.c_str());
			for(int i = 2; i < argc; i++)
            {
				try
				{
					CImg<double> image(argv[i]);
                	warp(image, argv[i]);
				}
				catch(CImgIOException e)
				{
					cout << "Cannot open file " << argv[i] << endl;
				}
            }
		}
		else if(part == "part2.2")
		{
//			warp(query_image);
//			cout << "after wrap" << endl;*/
			//=================================================
			//part 2.2			
			vector< CImg<double> > test_images;
			vector< CImg<double> > gray_images;  //in gray scale
			vector< vector<SiftDescriptor> > test_des;
			              
        	for(int i = 3; i < argc; i++){
	  			string test_file = argv[i];
	  			CImg<double> test_image(test_file.c_str());
	  			test_images.push_back(test_image);
	  			CImg<double> gray_image = test_image.get_RGBtoHSI().get_channel(2);
	 		 	//gray_images.push_back(gray_image);
				test_des.push_back(Sift::compute_sift(gray_image));
			}
    		srand (time(NULL));
    		float ratio = 0.6;
			int iteration = 10000;
			cout<<"ratio threshold: "<<ratio<<" iteration: "<<iteration<<endl;
		//vector< vector<SiftDescriptor> > test_des;
		//for( int i = 0; i < gray_images.size(); i++){
	  	//		test_des.push_back(Sift::compute_sift(gray_images[i]));
		//	} 

			CImg<double> input_image(query.c_str());
			CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
                  	vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
			CImg<double> query(2*input_image.width(),2*input_image.height(),1,3);
			query.fill(255);
			
			for(int i=0; i < input_image.width(); i++){
			  for(int j=0; j < input_image.height(); j++){
			    query(i,j,0,0) = input_image(i,j,0,0);
			    query(i,j,0,1) = input_image(i,j,0,1);
			    query(i,j,0,2) = input_image(i,j,0,2);
			  }
			}
			int j;		        
			for(int i =0; i < test_des.size(); i++){
                          CImg<double> blend = query;
			  CImg<double> output = query;
			  cout<<"------"<<endl;
			  cout<< "image: "<< i<<endl;
			  vector <sift_pairs> pairs;
			  compare_des(descriptors, test_des[i], ratio, pairs);
			  vector< sift_pairs> model = RANSAC(pairs,2,2,iteration);
			  CImg< double> inverse_proj_matrix = proj_transform(model);
                          string output_name = argv[i+3];
                          warp_image(output ,blend, test_images[i],inverse_proj_matrix, output_name );
			}
      	}
    	else
      		throw std::string("unknown part!");
  	}
  	catch(const string &err) {
    	cerr << "Error: " << err << endl;
  	}
	return 0;
}
