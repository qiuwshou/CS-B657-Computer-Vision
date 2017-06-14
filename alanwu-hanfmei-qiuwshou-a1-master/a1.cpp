#include "SImage.h"
#include "SImageIO.h"
#include <cmath>
#include "math.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include "DrawText.h"

using namespace std;
// some statements:
// 1) different thresholds are used for different images. 
//      on line 491~503, there are suggestions on the comments for choosing the threshold.
// 2) different templates for different images. 
//      we provide 2 sets of templates for testing music1.png and music4.png.
//      details about how to choose them are on line 480~482.
// 
// The simple image class is called SDoublePlane, with each pixel represented as
// a double (floating point) type. This means that an SDoublePlane can represent
// values outside the range 0-255, and thus can represent squared gradient magnitudes,
// harris corner scores, etc. 
//
// The SImageIO class supports reading and writing PNG files. It will read in
// a color PNG file, convert it to grayscale, and then return it to you in 
// an SDoublePlane. The values in this SDoublePlane will be in the range [0,255].
//
// To write out an image, call write_png_file(). It takes three separate planes,
// one for each primary color (red, green, blue). To write a grayscale image,
// just pass the same SDoublePlane for all 3 planes. In order to get sensible
// results, the values in the SDoublePlane should be in the range [0,255].
//

// Below is a helper functions that overlays rectangles
// on an image plane for visualization purpose. 

// Draws a rectangle on an image plane, using the specified gray level value and line width.
//
void overlay_rectangle(SDoublePlane &input, int _top, int _left, int _bottom, int _right, double graylevel, int width)

{
  for(int w=-width/2; w<=width/2; w++) {
    int top = _top+w, left = _left+w, right=_right+w, bottom=_bottom+w;

    // if any of the coordinates are out-of-bounds, truncate them 
    top = min( max( top, 0 ), input.rows()-1);
    bottom = min( max( bottom, 0 ), input.rows()-1);
    left = min( max( left, 0 ), input.cols()-1);
    right = min( max( right, 0 ), input.cols()-1);
      
    // draw top and bottom lines
    for(int j=left; j<=right; j++)
	  input[top][j] = input[bottom][j] = graylevel;
    // draw left and right lines
    for(int i=top; i<=bottom; i++)
	  input[i][left] = input[i][right] = graylevel;
  }
}

// DetectedSymbol class may be helpful!
//  Feel free to modify.
//
typedef enum {NOTEHEAD=0, QUARTERREST=1, EIGHTHREST=2} Type;
class DetectedSymbol {
public:
  int row, col, width, height;
  Type type;
  char pitch;
  double confidence;
};

// Function that outputs the ascii detection output file
void  write_detection_txt(const string &filename, const vector<struct DetectedSymbol> &symbols)
{
  ofstream ofs(filename.c_str());

  for(int i=0; i<symbols.size(); i++)
    {
      const DetectedSymbol &s = symbols[i];
      ofs << s.row << " " << s.col << " " << s.width << " " << s.height << " ";
      if(s.type == NOTEHEAD)
	ofs << "filled_note " << s.pitch;
      else if(s.type == EIGHTHREST)
	ofs << "eighth_rest _";
      else 
	ofs << "quarter_rest _";
      ofs << " " << s.confidence << endl;
    }
}

// Function that outputs a visualization of detected symbols
void  write_detection_image(const string &filename, const vector<DetectedSymbol> &symbols, const SDoublePlane &input)
{
  SDoublePlane output_planes[3];
  for(int i=0; i<3; i++)
    output_planes[i] = input;

  for(int i=0; i<symbols.size(); i++)
    {
      const DetectedSymbol &s = symbols[i];

      overlay_rectangle(output_planes[s.type], s.row, s.col, s.row+s.height-1, s.col+s.width-1, 255, 2);
      overlay_rectangle(output_planes[(s.type+1) % 3], s.row, s.col, s.row+s.height-1, s.col+s.width-1, 0, 2);
      overlay_rectangle(output_planes[(s.type+2) % 3], s.row, s.col, s.row+s.height-1, s.col+s.width-1, 0, 2);

      if(s.type == NOTEHEAD)
	{
	  char str[] = {s.pitch, 0};
	  draw_text(output_planes[0], str, s.row, s.col+s.width+1, 0, 2);
	  draw_text(output_planes[1], str, s.row, s.col+s.width+1, 0, 2);
	  draw_text(output_planes[2], str, s.row, s.col+s.width+1, 0, 2);
	}
    }

  SImageIO::write_png_file(filename.c_str(), output_planes[0], output_planes[1], output_planes[2]);
}



// The rest of these functions are incomplete. These are just suggestions to 
// get you started -- feel free to add extra functions, change function
// parameters, etc.

// Convolve an image with a separable convolution kernel
//
SDoublePlane convolve_separable(const SDoublePlane &input, const SDoublePlane &row_filter, const SDoublePlane &col_filter)
{
  SDoublePlane output_int(input.rows(), input.cols());
  SDoublePlane output(input.rows(), input.cols());

  const int KERNAL_ROW = 3;
  const int KERNAL_COL = 3;

  for (int i=0; i < input.rows()-2; i++)
    {for (int j=0; j < input.cols()-2; j++)
      {for (int m=0; m < row_filter.rows(); m++)
        {
            output_int[i][j] = output_int[i][j] + input[i+m][j] * row_filter[m][0];
         }
       }
    }

  for (int i=0; i < input.rows()-2; i++)
    {for (int j=0; j < input.cols()-2; j++)
      {for (int n=0; n < col_filter.rows(); n++)
        {
            output[i][j] = output[i][j] + input[i][j+n] * col_filter[n][0];
         }
       }
    }
  return output;
}  


// Convolve an image with a separable convolution kernel
//
SDoublePlane convolve_general(const SDoublePlane &input, const SDoublePlane &filter)
{
  SDoublePlane output(input.rows(), input.cols());
    for(int r=0; r<input.rows(); r++)
        for(int c=0; c<input.cols(); c++)
            output[r][c] = input[r][c];
        
    int fr=filter.rows();
    int fc=filter.cols();
    int k=(fr-1)/2;
    double convs=0;
    for(int x=1; x<input.rows()-1; x++)
    {
    for(int y=1; y<input.cols()-1; y++)
    {
    for(int i=0; i<fr; i++)
    {  
      for (int j=0; j<fc; j++)
           {
            convs=convs+filter[i][j]*input[x-(-k+i)][y-(-k+j)];
     }
    }
    output[x][y]=convs;
    convs=0;
    }
    }
for(int x=0; x<input.rows();x++)
{
output[x][0] = output[x][1];
output[x][input.cols()-1] = output[x][input.cols()-2];
}
for(int y=0; y<input.cols();y++)
{
output[0][y] = output[1][y];
output[input.rows()-1][y] = output[input.rows()-2][y];
}


  // Convolution code here
  
  return output;
}

// I found standardization function to normalize the scores to (0, 255) on website, just for better visualization of the scores.

SDoublePlane standardization(const SDoublePlane &input)
{
    SDoublePlane output(input.rows(), input.cols());
    double max_pixel = input[0][0];
    double min_pixel = input[0][0];
    for(int i=0; i<input.rows(); i++)
    for(int j=0; j<input.cols(); j++)
    {
        if(input[i][j] > max_pixel) max_pixel = input[i][j];
        if(input[i][j] < min_pixel) min_pixel = input[i][j];
    }
   
    for(int a=0; a<input.rows(); a++)
    for(int b=0; b<input.cols(); b++)
    {
        double x = input[a][b] - min_pixel;
        output[a][b] = x*(255.0/max_pixel);
    }
    return output;
}

/*
SDoublePlane sharpen(const SDoublePlane &input, double alpha)
{
    SDoublePlane output(input.rows(), input.cols());
    SDoublePlane temp(input.rows(), input.cols());
    SDoublePlane sharpen_filter(3,3);
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
          sharpen_filter[i][j] = -1;
    sharpen_filter[1][1] = 8;
    
    temp = standardization(convolve_general(input, sharpen_filter));

    vector<DetectedSymbol> blank;
    write_detection_image("test.png", blank, temp);
    for(int i=0; i<input.rows(); i++)
        for(int j=0; j<input.cols(); j++)
            output[i][j] = input[i][j] + alpha*temp[i][j];
    output = standardization(output);
    return output;
}

*/

//find musical symbols by distance between template and part of picture.
vector<DetectedSymbol> template_match_4(const SDoublePlane &input, const SDoublePlane &tmp, int symbol_type, double thresh)
{
    DetectedSymbol s;
    vector<DetectedSymbol> output;
    vector<DetectedSymbol> blank;
    SDoublePlane scores4(input.rows(),input.cols());
    SDoublePlane detected4(input.rows(),input.cols());
    for(int a=0;a<input.rows();a++)
        for(int b=0;b<input.cols();b++)
            {
            scores4[a][b] = 255;
            detected4[a][b] = input[a][b];
            }            

    double distance=0;
    for(int x=0;x<input.rows()-tmp.rows();x++)
{
    for(int y=0;y<input.cols()-tmp.cols();y++)
{
    for(int i=0;i<tmp.rows();i++)
{
    for(int j=0;j<tmp.cols();j++)
{
    int d = tmp[i][j]-input[x+i][y+j];
    if(d>=0) distance = distance+d;
    else distance = distance-d;
}
}
    scores4[x][y] = distance/(tmp.rows()*tmp.cols());
    distance = 0; 
}
}
int count = 0;
    for(int aa=0;aa<scores4.rows();aa++)
        for(int bb=0;bb<scores4.cols();bb++)
            {
        if(scores4[aa][bb]<=thresh)
        {
            DetectedSymbol s;
            s.row = aa;
            s.col = bb;
            s.width = tmp.cols();
            s.height = tmp.rows();
            s.type = (Type) symbol_type;
            s.confidence = (thresh-scores4[aa][bb])/thresh;
            if(symbol_type == 0) 
        {
    if(aa<input.rows()/2) s.pitch = (6-((int)(aa/6+4)%7)) + 'A';
    else s.pitch= (6-((int)aa/6+3)%7) + 'A';
        }
            output.push_back(s);
        }
            }
    
    write_detection_image("scores4.png", blank, scores4);

    return output;    
}

// Apply a sobel operator to an image, returns the result
// 
SDoublePlane sobel_gradient_filter(const SDoublePlane &input, bool _gx)
{
  SDoublePlane output(input.rows(), input.cols());
  double a[9]={1,2,1,0,0,0,-1,-2,-1};
  SDoublePlane sx_filter(3,3);
  SDoublePlane sy_filter(3,3);
  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    {  sx_filter[i][j]=-a[i+j*3]/8.0;
      sy_filter[i][j]=a[i*3+j]/8.0;
    }
    if(_gx == true) output = convolve_general(input, sx_filter);
    else output = convolve_general(input, sy_filter);
  // Implement a sobel gradient estimation filter with 1-d filters
  
  return output;
}


// Apply an edge detector to an image, returns the binary edge map
// 
SDoublePlane find_edges(const SDoublePlane &input, double thresh)
{
  SDoublePlane output(input.rows(), input.cols());
  SDoublePlane filtered_x = sobel_gradient_filter(input, true);
  SDoublePlane filtered_y = sobel_gradient_filter(input, false);
  for(int i=0; i<output.rows(); i++)
  for(int j=0; j<output.cols(); j++)
{
    double mag = filtered_x[i][j]*filtered_x[i][j] + filtered_y[i][j]*filtered_y[i][j];
    if (sqrt(mag)>thresh) output[i][j] = 255;
    else output[i][j] = 0;
}
  // Implement an edge detector of your choice, e.g.
          // use your sobel gradient operator to compute the gradient magnitude and threshold
  return output;
}

vector<DetectedSymbol> template_match_5(const SDoublePlane &input, const SDoublePlane &tmp, const int symbol_type, double thresh)
{
    vector<DetectedSymbol> output;
    SDoublePlane scores5(input.rows(),input.cols());
    SDoublePlane detected5(input.rows(),input.cols());
    for(int a=0;a<input.rows();a++)
        for(int b=0;b<input.cols();b++)
            {
            scores5[a][b]=255;
            detected5[a][b] = input[a][b];
            }

    double score = 0;
    SDoublePlane image_edges = find_edges(input, 20);
    write_detection_image("edges.png", output, image_edges);
    SDoublePlane tmp_edges = find_edges(tmp, 40);
    SDoublePlane D_to_edge(input.rows(), input.cols());

for(int a=0;a<image_edges.rows();a++)
for(int b=0;b<image_edges.cols();b++)  
    {
        if(image_edges[a][b] == 255) image_edges[a][b] = 1;
        else image_edges[a][b] = 0;
    }

for(int a=0;a<image_edges.rows();a++)
for(int b=0;b<image_edges.cols();b++)  
    {
    double distance = 1000;
    D_to_edge[a][b] = 1000;
    int up = a-20;
    int left = b-20;
    int down = a+20;
    int right = b+20;
    if(up<0) up = 0;
    if(left<0) left = 0;
    if(down>image_edges.rows()-1) down = image_edges.rows()-1;
    if(right>image_edges.cols()-1) right = image_edges.cols()-1;
    double d;
    for(int i=up; i<down; i++)
    for(int j=left; j<right; j++) 
    {
        if(image_edges[i][j]==1)
        {
        d = sqrt(pow(a-i,2) + pow(b-j,2));
        if(d<distance) distance = d; 
        }
    }
    D_to_edge[a][b] = distance;
    }


for(int a=0;a<tmp_edges.rows();a++)
for(int b=0;b<tmp_edges.cols();b++)  
    {if(tmp_edges[a][b] == 255) 
        {
         tmp_edges[a][b] = 1;
        }

    }
double pixel_score = 0;

for(int a=0; a<image_edges.rows()-tmp_edges.rows();a++)
for(int b=0; b<image_edges.cols()-tmp_edges.cols();b++)
{
    for(int i = 0; i<tmp_edges.rows();i++)
    for(int j = 0; j<tmp_edges.cols();j++)
    {
        pixel_score = pixel_score + D_to_edge[a+i][b+j]*tmp_edges[i][j];
    }
scores5[a][b] = pixel_score;
pixel_score = 0;
}

for(int a=0; a<scores5.rows()-tmp_edges.rows();a++)
for(int b=0; b<scores5.cols()-tmp_edges.cols();b++)
{
   if(scores5[a][b]<thresh) 
   {
            
            DetectedSymbol s;
            s.row = a;
            s.col = b;
            s.width = tmp.cols();
            s.height = tmp.rows();
            s.type = (Type) symbol_type;
            s.confidence = (thresh - scores5[a][b])/thresh;
            if(symbol_type == 0)// s.pitch = (rand() % 7) + 'A';
        {
    if(a<input.rows()/2) s.pitch = (6-((int)(a/6+4)%7)) + 'A'; //
    else s.pitch= (6-((int)a/6+3)%7) + 'A';
        }
            output.push_back(s);
    }

}

    return output;
}



SDoublePlane hough_transfer(const SDoublePlane &input)
{
    SDoublePlane vote_space(input.rows(), input.rows()/8);
    SDoublePlane edges = find_edges(input, 1);
    for(int i=0; i<edges.rows(); i++)
    for(int j=0; j<edges.cols(); j++)
    {
        for(int y=3; y<vote_space.cols(); y++)
        {
            for(int k=0; k<5; k++)
                {
                    if(i+k*y<edges.rows()) vote_space[i][y] += edges[i+k*y][j];
                }
        }
    }
    return vote_space;    
}

// This main file just outputs a few test images. You'll want to change it to do 
//  something more interesting!
//
int main(int argc, char *argv[])
{
  if(!(argc == 2))
    {
      cerr << "usage: " << argv[0] << " input_image" << endl;
      return 1;
    }
  
  vector<DetectedSymbol> blank;
  string input_filename(argv[1]);
  string template_filename1 = "template1.png";//template1.png for music1.png, template1_4.png for music_4.
  string template_filename2 = "template2.png";//same as above
  string template_filename3 = "template3.png";//same as above
  SDoublePlane input_image= SImageIO::read_png_file(input_filename.c_str());
  SDoublePlane symbol_template1= SImageIO::read_png_file(template_filename1.c_str());
  SDoublePlane symbol_template2= SImageIO::read_png_file(template_filename2.c_str());
  SDoublePlane symbol_template3= SImageIO::read_png_file(template_filename3.c_str());
  // test step 2 by applying mean filters to the input image
  
  //  with your symbol detection code obviously!
  SDoublePlane blank_image(input_image.rows(), input_image.cols());
  vector<DetectedSymbol> sss2 = template_match_4(input_image, symbol_template2, 1, 50.0);//
  vector<DetectedSymbol> sss3 = template_match_4(input_image, symbol_template3, 2, 23.0);//23 for m1, 58 for m2, 60 for m4
  vector<DetectedSymbol> sss1 = template_match_4(input_image, symbol_template1, 0, 23.0);//23 for m1, 45 for m2, 50 for m4
  vector<DetectedSymbol> total_4;
  for(int i=0; i<sss1.size();i++) total_4.push_back(sss1[i]);
  for(int i=0; i<sss2.size();i++) total_4.push_back(sss2[i]);
  for(int i=0; i<sss3.size();i++) total_4.push_back(sss3[i]);
  write_detection_image("detected4.png", total_4, input_image);
  

  vector<DetectedSymbol> ss1= template_match_5(input_image, symbol_template1,0, 10.0);//10 for m1, 50 for m4
  vector<DetectedSymbol> ss2= template_match_5(input_image, symbol_template2,1, 10.0);//10 for m1, 50 for m4
  vector<DetectedSymbol> ss3= template_match_5(input_image, symbol_template3,2, 50.0);
  vector<DetectedSymbol> total_5;
  for(int i=0; i<ss1.size();i++) total_5.push_back(ss1[i]);
  for(int i=0; i<ss2.size();i++) total_5.push_back(ss2[i]);
  for(int i=0; i<ss3.size();i++) total_5.push_back(ss3[i]);
  write_detection_image("detected5.png", total_5, input_image);
  


  SDoublePlane output(input_image.rows(),input_image.cols());
  for(int i=0;i<input_image.rows();i++)
  for(int j=0;j<input_image.cols();j++)
     output[i][j] = input_image[i][j];

  
  SDoublePlane hough = hough_transfer(input_image);
  hough = standardization(hough);
  write_detection_image("hough.png", blank,hough);

//  SImageIO::write_png_file("input.png", input,input,input);

  for(int x=0; x<hough.rows(); x++)
  for(int y=0; y<hough.cols(); y++)
    {
        if(hough[x][y]>230)
        {
            for(int k=0; k<5; k++)
            overlay_rectangle(output, x+k*y, 0, x+k*y, input_image.cols(), 255,2);
        }
    }
  SImageIO::write_png_file("staves.png", input_image, input_image, output);

write_detection_image("detected7.png", total_4, input_image);
write_detection_txt("detected7.txt", total_4);
}
