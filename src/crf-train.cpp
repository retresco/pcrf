////////////////////////////////////////////////////////////////////////////////////////////////////
// crf-train.cpp
// TH, June 2015
// Reads a training corpus file and trains a linear CRF model on it
// Format of the training corpus: 
//    corpus    = instance*
//    instance  = line+ newline
//    line      = label token attr*
// label, token and attr are strings
// TODO:
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <tclap/CmdLine.h>

#include <boost/tokenizer.hpp>

#include "../include/SimpleLinearCRFModel.hpp"
#include "../include/CRFTrainingCorpus.hpp"
#include "../include/AveragedPerceptronCRFTrainer.hpp"
#include "../include/CRFUtils.hpp"

typedef std::vector<std::string>   StringVector;

/// Holds training parameters
struct CRFTrainingHyperParams
{
  unsigned order;
  unsigned num_iterations;
  CRFTrainingAlgorithm method;
}; // CRFTrainingHyperParams


// Prototypes
void parse_options(int argc, char* argv[], std::string&, std::string&, CRFTrainingHyperParams&);
void usage();
template<unsigned O> void train_with_perceptron(CRFTranslatedTrainingCorpus&, const CRFTrainingHyperParams&, const std::string&);
template<unsigned O> void write_model(const SimpleLinearCRFModel<O>&, std::string, std::string);


int main(int argc, char* argv[])
{
  CRFTrainingHyperParams hyper_params;
  std::string model_file, corpus_file;

  parse_options(argc, argv, model_file, corpus_file, hyper_params);

  if (hyper_params.order > 3) {
    std::cerr << "crf-train: Error: Currently, only the orders 1, 2 or 3 are supported" << std::endl;
    exit(2);
  }

  // Open corpus file
  std::ifstream corpus_in(corpus_file.c_str());
  if (!corpus_in) {
    std::cerr << "crf-train: Error: Unable to open training corpus file '" << corpus_file << "\n";
    exit(3);
  }
  
  time_t t_start = clock();
  std::cerr << "Reading training data ";
  CRFTranslatedTrainingCorpus corpus(corpus_in);
  std::cerr << "\n[" 
            << corpus.labels_count() << " labels, " 
            << corpus.attributes_count() << " attributes, " 
            << corpus.token_count() << " tokens, " 
            << corpus.size() << " sequences]\n";

  if (corpus.labels_count() > 1000) {
    std::cerr << "crf-train: Warning: The number of labels is unusually high. You may experience memory problems\n";
  }

  if (hyper_params.method == crfTrainAveragedPerceptron) {
    if (hyper_params.order == 1)      train_with_perceptron<1>(corpus,hyper_params,model_file);
    else if (hyper_params.order == 2) train_with_perceptron<2>(corpus,hyper_params,model_file);
    else if (hyper_params.order == 3) train_with_perceptron<3>(corpus,hyper_params,model_file);
  }
  else if (hyper_params.method == crfTrainSGDL2) {
  }
  else {
    std::cerr << "Error: crf-train: unknown algorithm\n";
    exit(1);
  }
  
  std::cerr << "Total time: " << (float(clock()-t_start)/CLOCKS_PER_SEC) << "s\n";
}


template<unsigned ORDER>
void train_with_perceptron(CRFTranslatedTrainingCorpus& corpus, 
                           const CRFTrainingHyperParams& hyper_params,
                           const std::string& model_file)
{
  std::cerr << "crf-train: training model with order=" << ORDER << std::endl;
  time_t t0 = clock();
  AveragedPerceptronCRFTrainer<ORDER> perceptron_trainer(corpus);
  perceptron_trainer.train_by_number_of_iterations(hyper_params.num_iterations);
  std::cerr << "Training time: " << (float(clock()-t0)/CLOCKS_PER_SEC) << "s\n";
  
  write_model(perceptron_trainer.get_model(),model_file,model_file+".text_model");
  //std::ofstream dot("model.dot");
  //perceptron_trainer.get_model().draw(dot);
  
  model_info(perceptron_trainer.get_model());
}


template<unsigned ORDER>
void write_model(const SimpleLinearCRFModel<ORDER>& crf_model, std::string binary_file_name, std::string text_file_name)
{
  std::cerr << "Writing binary model '" << binary_file_name << "'\n";
  std::ofstream model_out(binary_file_name.c_str(), std::ios::binary);
  crf_model.write_model(model_out);

  //std::ofstream text_model_out(text_file_name.c_str());
  //text_model_out << crf_model;
}

void parse_options(int argc, char* argv[], std::string& model_file, std::string& corpus_file, CRFTrainingHyperParams& hyper_params)
{
  typedef TCLAP::ValueArg<std::string>  StringValueArg;
  typedef TCLAP::SwitchArg              BoolArg;
  typedef TCLAP::ValueArg<unsigned>     IntValueArg;

  if (argc == 1) {
    usage();
  }

  try {
    TCLAP::CmdLine cmd("crf-train -- Applies a trained CRF model to a input textfile\n",' ',"1.0");
    StringValueArg model_file_arg("m","model","Binary model file",true,"","filename");
    StringValueArg algorithm_arg("a","algorithm","Training algorithm",false,"","{perceptron}");
    IntValueArg num_iterations_arg("n","num-iterations","Number of iterations",false,100,"positive integer");
    IntValueArg order_arg("o","order","Model order",false,1,"1,2 or 3");
    TCLAP::UnlabeledMultiArg<std::string> input_files_arg("input","input files",true,"input-filename");

    cmd.add(model_file_arg);
    cmd.add(num_iterations_arg);
    cmd.add(order_arg);
    cmd.add(input_files_arg);

    cmd.parse(argc,argv);

    model_file = model_file_arg.getValue();
    hyper_params.method = crfTrainAveragedPerceptron;
    hyper_params.num_iterations = num_iterations_arg.getValue();
    hyper_params.order = order_arg.getValue();
    corpus_file = input_files_arg.getValue()[0];
  }

  catch (TCLAP::ArgException &e) { // catch any exceptions
    std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
  }
} 


void usage()
{
  std::cerr << "Usage: " << "crf-train" << " -m MODEL-FILE [-n NUM-ITERATIONS] [-o MODEL-ORDER] CORPUS-FILE" << std::endl << std::endl;
  std::cerr << "  MODEL-FILE is the binary file containing the trained model" << std::endl;
  std::cerr << "  CORPUS-FILE is a tab separated file containing a single sequence element per line" << std::endl;
  std::cerr << "    The format of each line is the following: OUTPUT-LABEL TOKEN FEAT1 FEAT2 ..." << std::endl;
  std::cerr << "    Different sequences are separated by an empty line" << std::endl;
  std::cerr << "  -n specifies the number of iterations\n";
  std::cerr << "  -o specifies the order of the model (1,2 or 3)\n";
  std::cerr << std::endl << "Example: crf-train -m mymodel.crf my.corpus" << std::endl;
  exit(1);
}

