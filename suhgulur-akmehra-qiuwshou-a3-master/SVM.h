/* Baseline SVM Class
Used to run baseline SVM mdel
Image 40X40 , color and black and white training
*/


class SVM : public Classifier
{
public:
  SVM(const vector<string> &_class_list) : Classifier(_class_list) 
  {
	  //Map for classname to class number
	  for(int i = 0;i<_class_list.size();++i)
	  {
		_classNumber[_class_list[i]] = i+1;	
	  }
  }
  
  /*
	Overloaded function to train baseline SVM classifier
  */
  virtual void train(const Dataset &filenames) 
  {
	/*Files needed to send data to Cornell SVM multiclass API*/  
	string SVMTrainData = "SVM/SVMTrainData.txt";
	string SVMModel = "SVM/SVMModel.txt";
	ofstream trainData(SVMTrainData.c_str());  
	
	//For each class
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
    {
		cout << "Processing " << c_iter->first << endl;
		
		// Get the features as required for Cornell SVM multiclass for each image of the class
		for(int i=0; i<c_iter->second.size(); i++)
		{
			//Class number from the class name
			trainData<<_classNumber[c_iter->first.c_str()];
			//Extract unrolled image size X size into 1 X size*size
			CImg<double> train_image = extract_features(c_iter->second[i].c_str());
			//Write it into cornell SVM multiclass train data format 
			for(int w = 0;w<train_image.width();++w)
				trainData<<" "<<w+1<<":"<<train_image(w,0,0);
			trainData<<endl;
		}
    }
	
	/*
		Call the API with train data. Model is saved in SVM modal file
	*/
	string command = "SVM/svm_multiclass_learn -c 1.0 "+ SVMTrainData+ " " +SVMModel; 
	system(command.c_str());
  }

  /*
	Overloaded function which classifies individual files
  */
  virtual string classify(const string &filename)
  {
	//Preparing test file for Cornell SVM multiclass classsify API. 	
    string SVMTestData = "SVM/SVMTestData.txt";
	string SVMModel = "SVM/SVMModel.txt";
	string Prediction = "SVM/Prediction.txt";
	{
		ofstream testData(SVMTestData.c_str());
		testData<<1<<" "; //Random classnumber guessing.
		//Extract features of the test image
		CImg<double> test_image = extract_features(filename);
		//Write that into test file
		for(int w = 0;w<test_image.width();++w)
			testData<<" "<<w+1<<":"<<test_image(w,0,0);
		testData<<endl;
	}
	//Call the classify API of cornell SVM Multiclass
    string command = "SVM/svm_multiclass_classify "+ SVMTestData+ " " +SVMModel+ " "+Prediction; 
	system(command.c_str());
    
	//Read the prediction output from predicted file
	ifstream readPrediction(Prediction.c_str());
	int prediction;
	readPrediction>>prediction;
	//Get corresponding class name
	string category = getClass(prediction);
    
	//return class name
	return category;
  }

  //No need to load model as We do that directly in Classify function : SVMModel
  virtual void load_model()
  {
	
  }
protected:
  bool color;
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
		color = true;
		
		CImg<double> image(filename.c_str());
		
		if(!color)
		{
			CImg<double> gray = getGrayScale(image);
			return gray.resize(size,size,1,1).unroll('x');
		}	
		return image.resize(size,size,1,3).unroll('x');
    }

  static const int size=40;  // subsampled image resolution
  map<string,int> _classNumber;//Map to store class name to class number 
private: 

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
  
  // Function to convert image to grey scale 
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
};
