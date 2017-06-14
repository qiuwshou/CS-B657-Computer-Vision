#include <fstream>

class Classifier
{
public:
  Classifier(const vector<string> &_class_list) : class_list(_class_list) {}

  // Run training on a given dataset.
  virtual void train(const Dataset &filenames) = 0;

  // Classify a single image.
  virtual string classify(const string &filename) = 0;

  // Load in a trained model.
  virtual void load_model() = 0;

  // Loop through all test images, hiding correct labels and checking if we get them right.
  void test(const Dataset &filenames)
  {
    cerr << "Loading model..." << endl;
    load_model();

    // loop through images, doing classification
    map<string, map<string, string> > predictions;
    for(map<string, vector<string> >::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter) 
      for(vector<string>::const_iterator f_iter = c_iter->second.begin(); f_iter != c_iter->second.end(); ++f_iter)
	{
	  cerr << "Classifying " << *f_iter << "..." << endl;
	  predictions[c_iter->first][*f_iter]=classify(*f_iter);
	}
    
    // now score!
    map< string, map< string, double > > confusion;
    int correct=0, total=0;
    for(map<string, vector<string> >::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter) 
      for(vector<string>::const_iterator f_iter = c_iter->second.begin(); f_iter != c_iter->second.end(); ++f_iter, ++total)
	confusion[c_iter->first][ predictions[c_iter->first][*f_iter] ]++;
    
    cout << "Confusion matrix:" << endl << setw(20) << " " << " ";
    for(int j=0; j<class_list.size(); j++)
      cout << setw(2) << class_list[j].substr(0, 2) << " ";

    for(int i=0; i<class_list.size(); i++)
      {
	cout << endl << setw(20) << class_list[i] << " ";
	for(int j=0; j<class_list.size(); j++)
	  cout << setw(2) << confusion[ class_list[i] ][ class_list[j] ] << (j==i?".":" ");

	correct += confusion[ class_list[i] ][ class_list[i] ];
      }
    
    cout << endl << "Classifier accuracy: " << correct << " of " << total << " = " << setw(5) << setprecision(2) << correct/double(total)*100 << "%";
    cout << "  (versus random guessing accuracy of " << setw(5) << setprecision(2) << 1.0/class_list.size()*100 << "%)" << endl;
  }
  
protected:
  vector<string> class_list;
};
