/*
Part 1.1
PCA and training SVM
Acknowledgements : The procedure is taken from https://onionesquereality.wordpress.com/2009/02/11/face-recognition-using-eigenfaces-and-distance-classifiers-a-tutorial/
*/
class Eigen : public Classifier
{
public:
  Eigen(const vector<string> &_class_list) : Classifier(_class_list) 
  {
	  //Creating a map of class number and class name
	  for(int i = 0;i<_class_list.size();++i)
	  {
		_classNumber[_class_list[i]] = i+1;	
	  }
  }
  
  /*
	overriden method
  */
  virtual void train(const Dataset &filenames) 
  {
	//1600(40 X 40) X 1250 training images data matrix
	CImg<double> data(1250,1600,1,1,0);
	int j = 0;
	vector<int> classes;
	
	//For each class	
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
    {
		cout << "Processing " << c_iter->first << endl;
		
		//for each image of the class
		for(int i=0; i<c_iter->second.size(); i++)
		{	
			//Store the image class number
			classes.push_back(_classNumber[c_iter->first.c_str()]);
			
			//Extract Image features
			CImg<double> imageFeatures = extract_features(c_iter->second[i].c_str());
			//Add the imageData into the data column j(1250)
			addData(imageFeatures,data,j);
			j++;	
		}
    }
	
	/*ofstream off("Eigen/Image.txt");//Helper for printing imtermediate data
	for(int w = 0;w<data.width();++w)
	{
		for(int h = 0;h<data.height();++h)
		{
		  off<<data(w,h)<<" ";
		}
		off<<endl;
	}*/
	
	//Average Image
	CImg<double> avgImage(1,1600,1,1,0);
	CImg<double> copyData = data;
	
	//Get teh mean Image
	getAvgImage(data,avgImage);
	
	/*ofstream m("Eigen/avg.txt");//Helper for printing imtermediate data
	for(int h = 0;h< avgImage.height();++h)
	{
		for(int w =0;w<avgImage.width();++w)
		{
			m<<avgImage(w,h)<<" ";
		}
		m<<endl;
	}*/
	
	//Visulaize teh average image
	visualizeImage(avgImage,"Eigen/AverageImage.png");
	
	//Subtract the mean image from all the images
	//subtractAvgImageFromEveryImage(data,avgImage);
	data = data -(avgImage);
	
	/*ofstream of("Eigen/Sub.txt");//Helper for printing imtermediate data
	for(int w = 0;w<data.width();++w)
	{
		for(int h = 0;h<data.height();++h)
		{
		  of<<data(w,h)<<" ";
		}
		of<<endl;
	}*/
	
	//data is now phi
	
	//This is to do AATranspose
	CImg<double> phi = data;
	data.transpose();
	
	CImg<double> coFactorMatrix = phi*data;
	/*ofstream ofd("Eigen/Covariance.txt");////Helper for printing imtermediate data
	for(int w = 0;w<coFactorMatrix.width();++w)
	{
		for(int h = 0;h<coFactorMatrix.height();++h)
		{
		  ofd<<coFactorMatrix(w,h)<<" ";
		}
		ofd<<endl;
	}*/
	
	
	//Compute Eigen values and Eigen vectors
	CImg<double> eigenValues,eigenVectors;
	coFactorMatrix.symmetric_eigen(eigenValues,eigenVectors);
	
	/*ofstream Values("Eigen/Values.txt");//Helper for printing imtermediate data
	Values<<"Dimension = "<<eigenValues.height()<<" X "<<eigenValues.width()<<endl;
	for(int h = 0;h<eigenValues.height();++h)
		Values<<eigenValues(0,h)<<endl;
	
	ofstream Vectors("Eigen/Vectors.txt");
	Vectors<<"Dimension = "<<eigenVectors.height()<<" X "<<eigenVectors.width()<<endl;
	for(int h = 0;h<eigenVectors.height();++h)
	{
		for(int w = 0;w<eigenVectors.width();++w)
		{
			Vectors<<eigenVectors(w,h)<<" ";
		}
		Vectors<<endl;
	}*/
	
	//picking top k eigen faces
	int k = 100;
	
	//TOpk eigen vectors 
	CImg<double> topK(k,eigenVectors.height(),1,1,0);
	getTopK(eigenVectors,topK);// 1600 X K
	
	//Pring eigen faces
	for(int w = 0;w<k;++w)
	{
		viewEigenface(eigenVectors,w);
	}
	
	topK.transpose();
	
	//Save Vectors for doing test
	ofstream model("Eigen/Eigen_model.txt");
	model<< topK.height() << " " << topK.width() << endl;
	for(int h = 0;h< topK.height();++h)
	{
		for(int w =0;w<topK.width();++w)
		{
			model<<topK(w,h)<<" ";
		}
		model<<endl;
	}

	//preparing data for SVM
	CImg<double> newData = topK*copyData;
	
	newData.transpose();
	
	//Saving for cornell SVM multiclass classifier
	string EigenTrainData = "Eigen/EigenTrainData.txt";
	string EigenModel = "Eigen/EigenModel.txt";
	ofstream trainData(EigenTrainData.c_str());
	
	for(int h = 0;h<newData.height();++h)
	{	
		//Saved class label
		trainData<<classes[h];
		for(int w = 0;w<newData.width();++w)
		{
			trainData<<" "<<w+1<<":"<<newData(w,h);
		}
		trainData<<endl;
	}
	
	//Call SVM command
	string command = "SVM/svm_multiclass_learn -c 1.0 "+ EigenTrainData+ " " +EigenModel; 
	system(command.c_str());
  }

  /*
	overridden method to classify individual images
  */
  virtual string classify(const string &filename)
  {
	string EigenTestData = "Eigen/EigenTestData.txt";
	string EigenModel = "Eigen/EigenModel.txt";
	string Prediction = "Eigen/Prediction.txt";
	CImg<double> copyData(1,1600,1,1,0);
	
	//Preparing test data from Cornell SVM classifier
	ofstream testData(EigenTestData.c_str());
	testData<<1<<" "; //Guessed class.
	
	//Extract the image features
	CImg<double> test_image = extract_features(filename);
	
	//Preparing data for matrix multiplication of eigen vectors and data extracted
	int c = 0;
	for(int h = 0;h<test_image.height();++h)
	{
		for(int w = 0;w<test_image.height();++w)
		{
			copyData(0,c++) = test_image(w,h,0);
		}
	}
	/*ofstream trainD("Eigen/model.txt");////Helper for printing imtermediate data
	for(int h = 0;h<model.height();++h)
	{	
		for(int w = 0;w<model.width();++w)
		{
			trainD<<" "<<w+1<<":"<<model(w,h);
		}
		trainD<<endl;
	}*/
	
	//Product of new vectors and data
	CImg<double> newData = model*copyData;
	
	
	//Preparing data for Cornell SVM multi classifer
	for(int h = 0;h<newData.height();++h)
		testData<<" "<<h+1<<":"<<newData(0,h,0);
	testData<<endl;
	
	//Call SVM classifier command
    string command = "SVM/svm_multiclass_classify "+ EigenTestData+ " " +EigenModel+ " "+Prediction; 
	system(command.c_str());
    
	//Read the prediction from predicted file
	ifstream readPrediction(Prediction.c_str());
	int prediction;
	readPrediction>>prediction;
	
	
	//get class name from thr class number
    string category = getClass(prediction);
	//Return category
	return category;
  }

  /*
	Loading eigen vectors saved fro testing
  */
  virtual void load_model()
  {
	int height,width;
	ifstream model_read("Eigen/Eigen_model.txt");
	
	model_read>>height>>width;
	CImg<double> read(width,height,1,1,0);
	for(int h = 0;h<read.height();++h)
	{
		for(int w =0;w<read.width();++w)
		{
			double d;
			model_read>>d;
			read(w,h) = d;
		}
	}
	model = read;
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.  
  CImg<double> extract_features(const string &filename)
    {
		//Load Image
		CImg<double> image(filename.c_str());
		//Convert to greyscale
		CImg<double> gray = getGrayScale(image);
		//CImg<double> gray = image.get_RGBtoYCbCr().get_channel(0);
		
		//Resize it 40 X 40
		CImg<double> img = gray.resize(size,size,1,1);
		return img;
    }

  static const int size=40;  // subsampled image resolution
  map<string,int> _classNumber; //Map to keep class number and class name matches
private:
	
  /*
	Code to convert color to Grey Scale 
  */
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
  
  /*
	Adding each image as a column of 40x40 features. adding to paricular column. j 0<=j<=1250
  */
  void addData(CImg<double>& imageFeatures,CImg<double>& data,int j)
  {
	int k = 0;
	for(int h = 0;h<imageFeatures.height();++h)
	{	
		for(int w = 0;w<imageFeatures.width();++w)
		{
			data(j,k++) = imageFeatures(w,h);
		}
	}
	
  }
  
  //finding avearge of all features 40X40. A column vector
  void getAvgImage(CImg<double>& data,CImg<double>& avgImage)
  {
	  for(int h = 0;h<data.height();++h)
	  {
		  double sum = 0.0;
		  for(int w = 0;w<data.width();++w)
		  {
			  sum+=data(w,h);
		  }
		  avgImage(0,h) = sum/data.width();
	  }
  }
  
  //TO visulaize a unrolled image as 40X40 image
  void visualizeImage(CImg<double>& avgImage,string name)
  {
	CImg<double> img(40,40,1,1,0);
	int k = 0;
	
	for(int h = 0;h<img.height();++h)
	{	
		for(int w = 0;w<img.width();++w)
		{
			img(w,h) = avgImage(0,k++);
		}
	}
	img.save_png(name.c_str());
	  
  }
  
  //To subtract average image from each image save as individual columns
  void subtractAvgImageFromEveryImage(CImg<double>& data,CImg<double>& avgImage)
  {
	for(int h = 0;h<data.height();++h)
	{	
		for(int w = 0;w<data.width();++w)
		{
			data(w,h) = data(w,h) - avgImage(0,h);
		}
	} 
  }
  
  //To get top k vectors from eigen vectors
  void getTopK(CImg<double>& vectors,CImg<double>& topK)
  {
	for(int h = 0;h<topK.height();++h)
	{	
		for(int w = 0;w<topK.width();++w)
		{
			topK(w,h) = vectors(w,h);
		}
	}  
  }
  
  //To unroll eigen vector and Save it as image(Eigen faces)
  void viewEigenface(CImg<double>& eigenVectors,int ww)
  {
	CImg<double> img(40,40,1,1,0);
	int k = 0;
	
	for(int h = 0;h<img.height();++h)
	{	
		for(int w = 0;w<img.width();++w)
		{
			img(w,h) = eigenVectors(ww,k++);
		}
	}
	img.normalize(0,255);
	string name = "Eigen/EigenFace"+to_string(ww)+".png";
	img.save_png(name.c_str());  
  }
  
	CImg<double> model; //Loaded Eigen vectors for testing
	
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
