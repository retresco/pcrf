
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "../include/SimpleLinearCRFModel.hpp"

#ifndef MODEL_ORDER
  #define MODEL_ORDER 1
#endif


template<unsigned O>
void model_info(const SimpleLinearCRFModel<O>& crf_model)
{
  std::cerr << "\n============================================\n";
  std::cerr << "Model information\n";
  std::cerr << "============================================\n";
  std::cerr << "# labels:      " << crf_model.labels_count() << "\n";
  std::cerr << "# transitions: " << crf_model.transitions_count() << "\n";
  std::cerr << "# features:    " << crf_model.features_count() << "\n";
  std::cerr << "# attributes:  " << crf_model.attributes_count() << "\n";
  std::cerr << "# parameters:  " << crf_model.parameters_count() << "\n";
  
  const ParameterVector& p = crf_model.get_parameters();
  unsigned nn = 0;
  for (unsigned i = 0; i < p.size(); ++i) {
    if (p[i] != 0.0) ++nn;
  }
  std::cerr << "  # non-null parameters: " << nn << "\n";
  std::cerr << "============================================\n\n";
}


int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: crf-test CRFSUITE-MODEL-FILE BINARY-MODEL-FILE" << std::endl;
    exit(1);
  }

  std::ifstream model_in(argv[1]);
  if (!model_in) {
    std::cerr << "Error: crf-convert: Could not open model file '" << argv[1] << "'" << std::endl;
    exit(2);
   }

  std::cerr << "Reading text model ...";
  SimpleLinearCRFModel<MODEL_ORDER> crf_model(model_in,false);
  std::cerr << " done" << std::endl;
  model_info(crf_model);

  std::cerr << "Writing binary model ...";
  std::ofstream model_out(argv[2], std::ios::binary);
  crf_model.write_model(model_out);
  std::cerr << " done" << std::endl;
  //std::ofstream model_out2("backup");
  //model_out2 << crf_model;
}
