#include<sstream>
#include<time.h>
typedef map<string,vector<int>> Bagword;
class Bags : public Classifier
{
public:
  Bags(const vector<string> &_class_list) : Classifier(_class_list) 
  {
		for(int i = 0;i<_class_list.size();++i)
			cout<<_class_list[i]<<endl;
	  for(int i = 0;i<_class_list.size();++i)
	  {
		_classNumber[_class_list[i]] = i+1;
	  }
  }
  
  
  virtual void train(const Dataset &filenames) 
  {     
    srand(time(NULL));
    clock_t t;
    t = clock();
	string SVMTrainData = "BOW/BOWTrainData_Sift.txt";
	string SVMModel = "BOW/BOWModel.txt";
	ofstream trainData(SVMTrainData.c_str());
        Bagword k_cluster = k_mean(filenames);
        int k;
	string c;
	vector<string> sep;

       
	for(Bagword::iterator c_iter=k_cluster.begin(); c_iter != k_cluster.end(); ++c_iter)
    {          
                sep = split(c_iter->first, '/');
		k = _classNumber[sep[1]];
		cout << "Processing " << c_iter->first << endl;
		trainData<<k;
		// Get the features as required for Cornell SVM multiclass
		for(int i=0; i<c_iter->second.size(); i++)
       		  trainData<<" "<<i+1<<":"<<c_iter->second[i];
                trainData<<endl;
    }
	string command = "SVM/svm_multiclass_learn -c 1.0 "+ SVMTrainData+ " " +SVMModel; 
	system(command.c_str());
	t = clock() - t;
	printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
  }

  virtual string classify(const string &filename)
  {
    srand(time(NULL));
    string BOWTestData = "BOW/BOWTestData_Sift.txt";
	string BOWModel = "BOW/BOWModel.txt";
	string Prediction = "BOW/Prediction_Sift.txt";
	//no need to initialize the centroids again
        //Bagword k_clust = k_mean(filanames, centroids, 1);
        ofstream testData(BOWTestData.c_str());
        vector<string> sep = split(filename,'/');
        string c  = sep[1];
        vector< vector<float>> centroids;
        CImg<double> centroids_image("BOW/centroids.png");
	for(int i=0; i<centroids_image.height();i++){
	  vector<float> f(128);
	  for(int j=0; j<centroids_image.width();j++){
	    f[j] = centroids_image(j,i,0,0);
	  }
          centroids.push_back(f);
	}

	{ 
          cout<<filename<<endl;
	  ofstream testData(BOWTestData.c_str());
	  testData<<_classNumber[c]<<" "; //Not sure why we put this but needed.
	  vector<int> features = get_features(filename,centroids);
	  for(int w = 0; w<features.size();++w)
	    testData<<" "<<w+1<<":"<<features[w];
	  testData<<endl;
	}


    string command = "SVM/svm_multiclass_classify "+ BOWTestData+ " " +BOWModel+ " "+Prediction; 
	system(command.c_str());
    
	ifstream readPrediction(Prediction.c_str());
	int prediction;
	readPrediction>>prediction;
	string category = getClass(prediction);
    
	return category;
  }

  virtual void load_model()
  {
	
  }
  
protected:
  CImg<double> extract_features(const string &filename)
    {
      CImg<double> image(filename.c_str());
      //CImg<double> gray = getGrayScale(image);
      
      return image.resize(size,size,1,3).unroll('x');
    }
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  static const int size=40;  // subsampled image resolution
  static const int iter= 10;
  //vector< vector<float> > centroids;
  map<string,int> _classNumber;
  //typedef map<string, vector<SiftDescriptor> > descriptor;
  Bagword k_mean(const Dataset filenames){
    Bagword bag_words;
    map<string, vector<SiftDescriptor> >records;
    vector< vector<float> >centroids;
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
        
 
	//create a map<imagename, sift of the image>
	for(int i=0;i< c_iter->second.size();i++){
	  CImg<double> input(c_iter->second[i].c_str());	  
          input.resize(140,140,1,3);
          CImg<double> gray = input.get_RGBtoHSI().get_channel(2);
          vector<SiftDescriptor> d = Sift::compute_sift(gray);
          records[c_iter->second[i]] = d;
	}
        vector<int> id_record;
        //initialize 450 features without repetition
	//a feature is a 128-d vector
	for(int j=0; j< 18; j++){  
        int c_id = rand() % c_iter->second.size();
	while( std::find(id_record.begin(),id_record.end(), c_id) != id_record.end()){
	  c_id = rand() % c_iter->second.size();
	}
	CImg<double>centroid(c_iter->second[c_id].c_str());
        centroid.resize(140,140,1,3);
        CImg<double> gray = centroid.get_RGBtoHSI().get_channel(2);
	vector<SiftDescriptor> d = Sift::compute_sift(gray);
        int d_id = rand() % d.size();
	centroids.push_back(d[d_id].descriptor);
	}	
      }
     
    //for each descriptor of image find the cloest feature
    //after that, update the representation of each feature
      for(int i=0; i<iter; i++){
        find_update(centroids, records);
      }
    
    
      //after the iteration, represent each image by 450 features 
    for(map<string, vector<SiftDescriptor> >::iterator it=records.begin(); it!=records.end();++it){
      vector<int> words_count(centroids.size(),0);
      for(int i=0; i<it->second.size();i++){
	words_count[it->second[i].row] += 1;
      }
      bag_words[it->first] = words_count;
    }

    //compute the final information of each feature
    //it will be used for test data
    CImg<double> centroids_image(128,centroids.size(),1,1,0);
    for(int i=0; i<centroids.size();i++){
      vector<float> f = centroids[i];
      for(int j=0; j<128;j++){
        //cout<<i<<"  "<<j<<endl;
	centroids_image(j,i,0,0) = f[j];
      }
    }
    //save the information of each features tha will be used for test data
    centroids_image.save("BOW/centroids.png"); 
    
    return bag_words;
  }

  void find_update(vector< vector<float> > &centroids, map<string, vector<SiftDescriptor> > &records){
    vector<int> count(centroids.size(),0);
    //for each descriptor find its cloest feature
    for(map<string, vector<SiftDescriptor> >::iterator it=records.begin(); it!=records.end();++it){
       for(int i = 0; i<it->second.size();i++){
	 double min_dist = 99999999;
         int group;
         vector<float> f = it->second[i].descriptor;
         for(int j=0; j<centroids.size(); j++){
	   double d = dist(centroids[j], f);
           if(d < min_dist){
             group = j;
             min_dist = d;   
	   }
	 }
	 //use the row information to save the id of feature
         vector<SiftDescriptor> d = records[it->first];
         d[i].row = group;
	 records[it->first] = d;
         count[group] += 1;
       } 
     }

    //count the reptition of each feature  
     for(map<string, vector<SiftDescriptor> >::iterator it=records.begin(); it!=records.end();++it){
       for(int i=0; i <it->second.size();i++){
	 int group = it->second[i].row;
	 for(int j=0; j<128; j++){
	   centroids[group][j] += it->second[i].descriptor[j];
	 }
       }
     }
     //update the value of features
     for(int i=0; i<centroids.size(); i++)
       for(int j=0; j< 128; j++)
	 centroids[i][j] = centroids[i][j] / count[i];
    
  }
  //calculate the ssd between two descriptors
  double dist(vector<float> &d1, vector<float> &d2){
    double sum = 0;
    for(int i =0; i < 128; i++){
      sum += pow(( d1[i] - d2[i]),2) ;
    }
    sum = sqrt(sum);
    return sum;
  }

  //reformat the test data basing on the information of features from training process
  vector<int> get_features(const string &filename, vector< vector<float> > &centroids){
    vector<int> f_count(centroids.size(),0);
    CImg<double> input(filename.c_str());
    input.resize(140,140,1,3);
    CImg<double> gray = input.get_RGBtoHSI().get_channel(2);
    vector<SiftDescriptor> d = Sift::compute_sift(gray);
    for(int i=0; i<d.size(); i++){
      int group;
      double min_dist = 9999999;
      for(int j=0;j<centroids.size();j++){
	double ssd = dist(centroids[j], d[i].descriptor);
	if(ssd < min_dist){
	  group = j;
	  min_dist = ssd;
	}
      }
      f_count[group] += 1;
    }
    return f_count;
  }
  //use this to split the filename. And use it to get the class name 
  //adopt from http://code.runnable.com/VHb0hWMZp-ws1gAr/splitting-a-string-into-a-vector-for-c%2B%2B
  vector<string> split(string str, char delimiter) {
    vector<string> internal;
    stringstream ss(str); // Turn the string into a stream.
    string tok;
  
    while(getline(ss, tok, delimiter)) {
      internal.push_back(tok);
    }
  
    return internal;
  }


private: 
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
  
  CImg<double> getGrayScale(CImg<double>& image)
  {
    CImg<double> gray(image.width(), image.height(), 1, 1,0);
	  
	for(int w = 0;w<image.width();w++)
	{
		for(int h = 0;h<image.height();h++)
		{
		  gray(w,h,0,0) = (image(w,h,0,0)+image(w,h,0,1)+image(w,h,0,2))/3;
		}
	}
		
	return gray;	
  }
  
};
