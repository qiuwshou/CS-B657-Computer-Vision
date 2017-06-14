/* CNN Class
*/

class CNN : public Classifier
{
public:
  CNN(const vector<string> &_class_list) : Classifier(_class_list) 
  {
	  //Map for classname to class number
	  for(int i = 0;i<_class_list.size();++i)
	  {
		_classNumber[_class_list[i]] = i+1;	
	  }
  }
  
  /*
	Overloaded function to train  SVM classifier
  */
  virtual void train(const Dataset &filenames) 
  {
	/*Files needed to send data to Cornell SVM multiclass API*/  
	string CNNTrainData = "CNN/CNNTrainData.txt";
	string CNNModel = "CNN/CNNModel.txt";
	ofstream trainData(CNNTrainData.c_str());  
	
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
		Call the API with train data. Model is saved in CNN modal file
	*/
	string command = "SVM/svm_multiclass_learn -c 1.0 "+ CNNTrainData+ " " +CNNModel; 
	system(command.c_str());
  }

  /*
	Overloaded function which classifies individual files
  */
  virtual string classify(const string &filename)
  {
	//Preparing test file for Cornell SVM multiclass classsify API. 	
    string CNNTestData = "CNN/CNNTestData.txt";
	string CNNModel = "CNN/CNNModel.txt";
	string Prediction = "CNN/Prediction.txt";
	{
		ofstream testData(CNNTestData.c_str());
		testData<<1<<" "; //Random classnumber guessing.
		//Extract features of the test image
		CImg<double> test_image = extract_features(filename);
		//Write that into test file
		for(int w = 0;w<test_image.width();++w)
			testData<<" "<<w+1<<":"<<test_image(w,0,0);
		testData<<endl;
	}
	//Call the classify API of cornell SVM Multiclass
    string command = "SVM/svm_multiclass_classify "+ CNNTestData+ " " +CNNModel+ " "+Prediction; 
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

  //No need to load model as We do that directly in Classify function : CNNModel
  virtual void load_model()
  {
	
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
		string CNNfeatures = "CNN/features.txt";
		string CNNImage_overfeat = "CNN/image_overfeat.jpg";
		
		CImg<double> image(filename.c_str());
		image.resize(size,size,1,3);//Keeping color characteristics
		
		//Saving the image as Overfeat needs that
		image.save(CNNImage_overfeat.c_str());
		
		//Execute Overfeat command to extract features 
		string command = "overfeat/bin/linux_64/overfeat -L 18 " + CNNImage_overfeat +" > "+CNNfeatures;
		system(command.c_str());
		
		int n,h,w;
		ifstream readFeatures(CNNfeatures.c_str());
		readFeatures>>n; 
		readFeatures>>h;
		readFeatures>>w;
		
		CImg<double> features(n*h*w,1);
		int k=0;
		while(!readFeatures.eof())
		{
			double d;
			readFeatures>>d;
			features(k++,0) = d;
		}
		return features;
		
	}

  static const int size=231;  // subsampled image resolution
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
};
