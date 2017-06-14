#include <string>

/*Custom classes to store the rectangular features
*/

/*Region class stores ends of rectangle*/
class region
{
	public:
	region(){}
	region(int _w1,int _h1,int _w2,int _h2):w1(_w1),h1(_h1),w2(_w2),h2(_h2)
	{}
	int w1,w2,h1,h2;
};

/*
	Feature class stores combination of white and black regions. costituting feature 
*/
class feature
{
	public:
	vector<region> blackRegions;
	vector<region> whiteRegions;
};

/*
	Haar Features viola jones implementation of classification. 
	//Acknowledge: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
	//Wikipedia: https://en.wikipedia.org/wiki/Summed_area_table
*/
class Haas : public Classifier
{
public:
  Haas(const vector<string> &_class_list) : Classifier(_class_list) 
  {
	  //Creating a map of class number and class name
	  for(int i = 0;i<_class_list.size();++i)
	  {
		_classNumber[_class_list[i]] = i+1;
	  }
  }
  
  /*
	Overloaded function for training SVM classifier
  */
  virtual void train(const Dataset &filenames) 
  { 
	//Files required for training SVM API  
	string HaasTrainData = "Haas/HaasTrainData.txt";
	string HaasModel = "Haas/HaasModel.txt";
	ofstream trainData(HaasTrainData.c_str());
	
	//Colled all 1600 different sizes of rectangle that can fit ina 40 X 40 images. The rectangle shapes are given by feature0
	//feature1 feature2 feature3. each of different scales
	vector<feature> Features = getFeatures();
	//Saving these recangles as it is needed to be extracted from test file
	save_features(Features);
	
	//Saving class numbers for each image
	vector<int> classes;
	
	int f = 0;	
	//Data for storing features extracted values
	CImg<double> requiredData(1000,1250,1,1,0);
	
	//For each class
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
    {
		cout << "Processing " << c_iter->first << endl;
		
		//For each image in that class
		for(int i=0; i<c_iter->second.size(); i++,++f)
		{	
			//Save class number
			classes.push_back(_classNumber[c_iter->first.c_str()]);
			
			//Extract features from each image
			CImg<double> imageFeatures = extract_features(c_iter->second[i].c_str());//.get_normalize(0,10);
			
			//Add the image features as Integral Image features using haar methods. Each in row f
			addFeatures(imageFeatures,Features,requiredData,f);
		}
    }
	
	//requiredData.normalize(0,10);
	//Writing the feature values in the file for Cornell SVM multiclass classifier traim API
	for(int h = 0;h<requiredData.height();++h)
	{
		//Class for each image which was saved
		trainData<<classes[h];
		//Writing all the feature for each image  in the file
		for(int w = 0;w<requiredData.width();++w)
		{
			trainData<<" "<<w+1<<":"<<requiredData(w,h);
		}
		trainData<<endl;
	}
	
	//Calling SVM classify command cornell multiclass SVM
	string command = "SVM/svm_multiclass_learn -c 1.0 "+ HaasTrainData+ " " +HaasModel; 
	system(command.c_str());
  }

  /*
	Overridden function to classify incoming new image file
  */
  virtual string classify(const string &filename)
  {
	/*Files for testing in cornell SVM multi class API*/	
	string HaasTestData = "Haas/HaasTestData.txt";
	string HaasModel = "Haas/HaasModel.txt";
	//test Data 
	CImg<double> requiredData(1000,1,1,1,0);
	
	string Prediction = "Haas/Prediction.txt";
	
		ofstream testData(HaasTestData.c_str());
				
		//Extract image pixels
		CImg<double> test_image = extract_features(filename);//.get_normalize(0,10);
		//Haar features created and saved as test data
		addFeatures(test_image,features,requiredData,0);
	
	
	//requiredData.normalize(0,10);
	
	//Test Data written in a file for SVM multiclass classifier  
	for(int h = 0;h<requiredData.height();++h)
	{
		testData<<1; //Randomly guessed test class.
		for(int w = 0;w<requiredData.width();++w)
		{
			testData<<" "<<w+1<<":"<<requiredData(w,h);
		}
		testData<<endl;
	}
	
	//Command to SVM multiclass API for classification
    string command = "SVM/svm_multiclass_classify "+ HaasTestData+ " " +HaasModel+ " "+Prediction; 
	system(command.c_str());
    
	//reading from prediction for knowing class
	ifstream readPrediction(Prediction.c_str());
	int prediction;
	readPrediction>>prediction;
	//string category = getClass(prediction);
	//Getting classname from class number
    string category = getClass(prediction);
	//Return category
	return category;
  }

  virtual void load_model()
  {
	  //Loading the saved rectangles features which is used in extracting features for test images
	  //Following is the coding pattern
	  //Eachline is one feature of reactangles. we have 4 values corresponding to w1 h1 w2 h2, A black region is first follwed by value 100 and then starts 
	  //White regions. If a pixel value after black or white rectangle is 255 then  it is end of that paricular feature. 
	  // if first value is 255 then it is end of features
	  CImg<int> Features("Haas/Features.png");
	  vector<feature> loadFeatures;
	  for(int h = 0;h<Features.height();++h)
	  {  
		if(Features(0,h) == 255)
			break;
		bool black = true;	
		feature f;
		for(int w = 0;w<Features.width();++w)
		{
			if(Features(w,h) == 100)
			{
				black = false;
				w++;
			}
			else if(Features(w,h) == 255)
				break;
			
			region r;
			r.w1 = Features(w++,h);	
			r.h1 = Features(w++,h);
			r.w2 = Features(w++,h);
			r.h2 = Features(w,h);
			if(black)
				f.blackRegions.push_back(r);
			else
				f.whiteRegions.push_back(r);
		}
		
		loadFeatures.push_back(f);
	  }

	/*ofstream ready("Haas/FeaturesRead.txt");//Helpers to see intermediate output
	
	for(int i = 0;i<loadFeatures.size();++i)
	{
		for(int j = 0;j<loadFeatures[i].blackRegions.size();++j)
		{
			ready<<"B"<<" "<<loadFeatures[i].blackRegions[j].w1<<" "<<loadFeatures[i].blackRegions[j].h1<<" "<<loadFeatures[i].blackRegions[j].w2<<" "<<loadFeatures[i].blackRegions[j].h2;	
		}
		for(int j = 0;j<loadFeatures[i].whiteRegions.size();++j)
		{
			ready<<"W"<<" "<<loadFeatures[i].whiteRegions[j].w1<<" "<<loadFeatures[i].whiteRegions[j].h1<<" "<<loadFeatures[i].whiteRegions[j].w2<<" "<<loadFeatures[i].whiteRegions[j].h2;			
		}
		ready<<endl;
	}*/
	
	//Save the read features/rectangles
	features = loadFeatures;

  }
protected:

  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
		CImg<double> image(filename.c_str());
		//Convert to grey scale
		CImg<double> gray = getGrayScale(image);
		//CImg<double> gray = image.get_RGBtoHSI().get_channel(2);
		//Resizing the image
		CImg<double> img = gray.resize(size,size,1,1);
		return img;
    }

  static const int size=40;  // subsampled image resolution
  map<string,int> _classNumber; //Class number to class name mapping
private:
  
  /* Code to convert to gray scale*/
  CImg<double> getGrayScale(CImg<double>& image)
  {
	CImg<double> gray(image.width(), image.height(), 1, 1, 0);
	  
	for(int w = 0;w<image.width();++w)
	{
		for(int h = 0;h<image.height();++h)
		{
		  gray(w,h,0,0) = (image(w,h,0,0)+image(w,h,0,1)+image(w,h,0,2))/3;
		}
	}	
	return gray;	
  }
  
  /* Extract 16000 rectangles*/
  vector<feature> getFeatures()
  {
	vector<feature> features;  
	int i = 0;
	while(i<1000)
	{
		//Create each feature based on one of the 4 random;y selected
		if(createFeature(features,rand()%4))
			++i;		
	}
	return features;	
  }
  
  /*
		this is feature 0:
		Pattern:                      
		                      ************
							  *		******		
							  *		******
							  *		******
							  *		******
							  *		******
							  *		******
							  ************
		                       
  */
  bool feature0(vector<feature>& features)
  {
	feature f1;
	//This feature is scalable in four sizes.
	
	int size = (rand() % 8) + 1;
	int w1 = rand() % 40; //Randomly chosen initial points
	int h1 = rand() % 40; //Randomly chosen initial points
	int w2 = w1 + (size * 1);
	int h2 = h1 + (size * 4);
	region r1(w1,h1,w2,h2);
	int w3 = w2;
	int h3 = h1;
	int w4 = w1 + (size * 2);
	int h4 = h2;
	region r2(w3,h3,w4,h4);
	//Only consider if it can fit into the 40 X 40 image
	if(w1 >= 40 || h1 >= 40 || w2 >= 40 || h2 >= 40 || w3 >= 40 || h3 >= 40 || w4 >= 40 || h4 >= 40)
		return false;
	else
	{
		f1.whiteRegions.push_back(r1);
		f1.blackRegions.push_back(r2);
		features.push_back(f1);
		return true;
	}  
  }
  
  /*
		this is feature 1:
		Pattern:                      
		                      ******************
							  *		     		*
							  *		      		*
							  *		    		*
							  *******************
							  *******************
							  *******************
							  *******************
		                       
  */
  bool feature1(vector<feature>& features)
  {
	feature f1;  
	
	//This feature is scalable in four sizes.
	int size = (rand() % 8) + 1;
	int w1 = rand() % 40;//Randomly chosen initial points
	int h1 = rand() % 40;//Randomly chosen initial points
	int w2 = w1 + (size * 1);
	int h2 = h1 + (size * 1);
	region r1(w1,h1,w2,h2);
	int w3 = w1;
	int h3 = h2;
	int w4 = w2;
	int h4 = h1 + (size * 2);
	region r2(w3,h3,w4,h4);
	//Only consider if it can fit into the 40 X 40 image
	if(w1 >= 40 || h1 >= 40 || w2 >= 40 || h2 >= 40 || w3 >= 40 || h3 >= 40 || w4 >= 40 || h4 >= 40)
		return false;
	else
	{
		f1.whiteRegions.push_back(r2);
		f1.blackRegions.push_back(r1);
		features.push_back(f1);
		return true;
	}
  }
  
  /*
		this is feature 2:
		Pattern:                      
		                      ******************
							  *     ******		*
							  *		******		*
							  *		******		*
							  *		******		*
							  *		******		*
							  *		******		*
							  *******************
							  
		                       
  */
  bool feature2(vector<feature>& features)
  {
	feature f1;  
	//This feature is scalable in four sizes. Randomly chosen
	int size = (rand() % 8) + 1;
	int w1 = rand() % 40; //Randomly chosen initial points
	int h1 = rand() % 40; //Randomly chosen initial points
	int w2 = w1 + (size * 1);
	int h2 = h1 + (size * 1);
	region r1(w1,h1,w2,h2);
	int w3 = w2;
	int h3 = h1;
	int w4 = w1 + (size * 2);
	int h4 = h2;
	region r2(w3,h3,w4,h4);
	int w5 = w4;
	int h5 = h1;
	int w6 = w1 + (size * 3);
	int h6 = h2;
	region r3(w5,h5,w6,h6);
	//Only consider if it can fit into the 40 X 40 image
	if(w1 >= 40 || h1 >= 40 || w2 >= 40 || h2 >= 40 || w3 >= 40 || h3 >= 40 || w4 >= 40 || h4 >= 40 || w5 >= 40 || h5>=40 || w6>=40 || h6 >=40)
		return false;
	else
	{
		f1.whiteRegions.push_back(r1);
		f1.whiteRegions.push_back(r3);
		f1.blackRegions.push_back(r2);
		features.push_back(f1);
		return true;
	}
  }
  
  /*
		this is feature 2:
		Pattern:                      
		                      ******************
							  *        *********
							  *		   *********
							  *		   *********
							  **********
							  **********
							  **********		   
							  ******************
							  
		                       
  */
  bool feature3(vector<feature>& features)
  {
	feature f1;  
	//This feature is scalable in four sizes. Randomly chosen
	int size = (rand() % 8) + 1;
	int w1 = rand() % 40;//Randomly chosen initial points
	int h1 = rand() % 40;//Randomly chosen initial points
	int w2 = w1 + (size * 1);
	int h2 = h1 + (size * 1);
	region r1(w1,h1,w2,h2);
	int w3 = w2;
	int h3 = h1;
	int w4 = w1 + (size * 2);
	int h4 = h2;
	region r2(w3,h3,w4,h4);
	int w5 = w1;
	int h5 = h2;
	int w6 = w2 ;
	int h6 = h1 + (size * 2);
	region r3(w5,h5,w6,h6);
	int w7 = w2;
	int h7 = h2;
	int w8 = w4;
	int h8 = h6;
	region r4(w7,h7,w8,h8);
	//Only consider if it can fit into the 40 X 40 image
	if(w1 >= 40 || h1 >= 40 || w2 >= 40 || h2 >= 40 || w3 >= 40 || h3 >= 40 || w4 >= 40 || h4 >= 40 || w5 >= 40 || h5>=40 || w6>=40 || h6 >=40 || w7>=40 || h7>=40 || w8>=40 || h8>=40)
		return false;
	else
	{
		f1.whiteRegions.push_back(r1);
		f1.whiteRegions.push_back(r4);
		f1.blackRegions.push_back(r2);
		f1.blackRegions.push_back(r3);
		features.push_back(f1);
		return true;
	}
  }
  
  /*
	Function to create features on randimly selected types
  */
  bool createFeature(vector<feature>& features,int type)
  {
	  switch(type)
	  {
		case 0: return feature0(features);
					
		case 1:	return feature1(features);
					
		case 2: return feature2(features);

		case 3: return feature3(features);	
	  };
  }
  
  /*
	Get Area specified by a region Given summed Area. Done Variation normalization to overcome intensity problems. 	
  */
  double getArea(region r,CImg<double>& summedArea,CImg<double>& summedSquaredArea)
  {
	double A = 0,AS =0;
    double B = 0,BS = 0;
    double C = 0,CS =0;
	int a = 0,b=0;
    if(r.w1 != 0)
	{
        B = summedArea(r.w1-1,r.h2);
		BS = summedSquaredArea(r.w1-1,r.h2);
		b = r.w1-1;
	}
    if(r.h1 != 0)
	{
        C = summedArea(r.w2,r.h1-1);
		CS = summedSquaredArea(r.w2,r.h1-1);
		a = r.h1-1;
    }
	if(r.w1 != 0 && r.h1 != 0)
	{
        A = summedArea(r.w1-1,r.h1-1);
		AS = summedSquaredArea(r.w1-1,r.h1-1);
	}
    double D = summedArea(r.w2,r.h2);
    //return (D-B-C+A);
	
	
	double areaValue = (D-B-C+A);  
	  
	double mean = areaValue/((r.w2-b) * (r.h2-a));
	  
	double sqrAreaValue = (summedSquaredArea(r.w2,r.h2) - BS - CS + AS);
	  
	int sqsd = (int)(sqrAreaValue/((r.w2-b) * (r.h2-a))) - pow(mean,2);
	int standardDeviation;
	if(sqsd > 0)
	  standardDeviation = sqrt(sqsd);
	else
		standardDeviation = 1;
	  
	double value = areaValue/standardDeviation;
	  
	return value;
  }
  
  //Given image pixels and features extract rectangle differences as expected and store it in feature image of image row j  
  void addFeatures(CImg<double>& imageFeatures,vector<feature>& features,CImg<double>& requiredData,int f)
  {
		//Integral Image of the image
		CImg<double> summedArea = getSummedArea(imageFeatures);
		//Integral Image of the squared images
		CImg<double> summedSquaredArea = getSummedSquaredArea(imageFeatures);
		
		//Haar features extraction
		for(int i = 0;i<features.size();++i)
		{
			double val = 0;
			//Adding black regions of the fetaure
			for(int j = 0;j<features[i].blackRegions.size();++j)
			{
				val+= getArea(features[i].blackRegions[j],summedArea,summedSquaredArea);
			}
			//Subtraction white regions of the feature
			for(int j = 0;j<features[i].whiteRegions.size();++j)
			{
				val-= getArea(features[i].whiteRegions[j],summedArea,summedSquaredArea);
			}
			requiredData(i,f) = val;				
		}
  }
  
  //Integral Image calculation summation area
  CImg<double> getSummedArea(CImg<double>& imageFeatures)
  {
	CImg<double> summedArea(imageFeatures.width(),imageFeatures.height(),1,1,0);
	
	summedArea(0,0) = imageFeatures(0,0);
	for(int w = 1;w<imageFeatures.width();++w)
		summedArea(w,0) = summedArea(w-1,0) + imageFeatures(w,0);
	
	for(int h = 1;h<imageFeatures.height();++h)
		summedArea(0,h) = summedArea(0,h-1) + imageFeatures(0,h);
	
	for(int h = 1;h<imageFeatures.height();++h)
	{
		for(int w = 1;w<imageFeatures.width();++w)
		{
			summedArea(w,h) = imageFeatures(w,h) + summedArea(w-1,h) + summedArea(w,h-1) - summedArea(w-1,h-1);
		}
	}
	return summedArea;	
  }
  
  /*Integral Image of squared Image* used for  Variation normalization*/ 
  CImg<double> getSummedSquaredArea(CImg<double>& imageFeatures)
  {
	CImg<double> summedSquaredArea(imageFeatures.width(),imageFeatures.height(),1,1,0);
	
	summedSquaredArea(0,0) = pow(imageFeatures(0,0),2);
	for(int w = 1;w<imageFeatures.width();++w)
		summedSquaredArea(w,0) = summedSquaredArea(w-1,0) + pow(imageFeatures(w,0),2);
	
	for(int h = 1;h<imageFeatures.height();++h)
		summedSquaredArea(0,h) = summedSquaredArea(0,h-1) + pow(imageFeatures(0,h),2);
	
	for(int w = 1;w<imageFeatures.width();++w)
	{
		for(int h = 1;h<imageFeatures.height();++h)
		{
			summedSquaredArea(w,h) = pow(imageFeatures(w,h),2) + summedSquaredArea(w-1,h) + summedSquaredArea(w,h-1) - summedSquaredArea(w-1,h-1);
		}
	}
	return summedSquaredArea;	
  }
  
  /*
	 method to save rectangles create. It is saves as PNG with pixel values coreesponding the rectangle points. Details present in load_model.
  */
  void save_features(vector<feature>& features)
  {
	//Temporary code
	//ofstream featurestxt("Haas/Features.txt");
	
	//maximum of 20000 rectangles
	CImg<int> Features(100,20000,1,1,255);
	int w = 0,h = 0;
	for(int i = 0;i<features.size();++i)
	{
		w = 0;
		//Encoding black regions
		for(int j = 0;j<features[i].blackRegions.size();++j)
		{
			//featurestxt<<"B"<<" "<<features[i].blackRegions[j].w1<<" "<<features[i].blackRegions[j].h1<<" "<<features[i].blackRegions[j].w2<<" "<<features[i].blackRegions[j].h2;
			
			Features(w++,h) = features[i].blackRegions[j].w1;
			Features(w++,h) = features[i].blackRegions[j].h1;
			Features(w++,h) = features[i].blackRegions[j].w2;
			Features(w++,h) = features[i].blackRegions[j].h2;
		}
		Features(w++,h) = 100;
		
		//Encoding white regions
		for(int j = 0;j<features[i].whiteRegions.size();++j)
		{
			//featurestxt<<"W"<<" "<<features[i].whiteRegions[j].w1<<" "<<features[i].whiteRegions[j].h1<<" "<<features[i].whiteRegions[j].w2<<" "<<features[i].whiteRegions[j].h2;
			
			Features(w++,h) = features[i].whiteRegions[j].w1;
			Features(w++,h) = features[i].whiteRegions[j].h1;
			Features(w++,h) = features[i].whiteRegions[j].w2;
			Features(w++,h) = features[i].whiteRegions[j].h2;
		}
		//featurestxt<<endl;
		h++;
	}
	//Save it as a PNG
	Features.save_png("Haas/Features.png");
	
	/*ofstream matrix("Haas/matrix.txt"); //Helper functions
	for(int h = 0;h<Features.height();++h)
	{
		for(int w = 0;w<Features.width();++w)
		{
			matrix<<Features(w,h)<<" ";
		}
		matrix<<endl;
	}*/
			
  }
  
  /* Helper fumction to retrieve class from the class number predicted
  */
  string getClass(int prediction)
  {
	  map<string,int>::iterator begin = _classNumber.begin();
	  map<string,int>::iterator end = _classNumber.end();
	  while(begin != end)
	  {
		  if(begin->second == prediction)
		  {
			  return begin->first;
		  }
		  ++begin;
	  }
	  return NULL;
  }

  //Object used to load saved rectangles.
  vector<feature> features;
};
