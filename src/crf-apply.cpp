
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <tclap/CmdLine.h>

#include "../include/CRFApplier.hpp"
#include "../include/SimpleLinearCRFModel.hpp"
#include "../include/CRFDecoder.hpp"
#include "../include/CRFUtils.hpp"
#include "../include/NERFeatureExtractor.hpp"
#include "../include/NERConfiguration.hpp"
#include "../include/AsyncTokenizer.hpp"
#include "../include/NEROutputters.hpp"
#include "../include/TokenWithTag.hpp"


// TODO: integrate this into the NERConfig
#define PERSONNAMES_LIST        "lists/personnames.list.bin"
#define NER_LIST_ALL            "lists/named-entities.list.bin"
#define NE_LEFTCONTEXT_LIST     "lists/lc.list.bin"
#define NE_RIGHTCONTEXT_LIST    "lists/rc.list.bin"
#define NE_REGEX_LIST           "lists/regex.list"


typedef std::vector<std::string>   StringVector;

// Prototypes
void parse_options(int argc, char* argv[], std::string&, StringVector&, NERConfiguration&, unsigned&, bool&, bool&, std::string&);
template<unsigned O> void load_and_apply_model(std::ifstream&,const std::string&,const StringVector&,const NERConfiguration&, bool, const std::string&);
template<unsigned O> void load_clue_lists(CRFApplier<O>&);
void usage();


int main(int argc, char* argv[])
{
  NERConfiguration ner_config;
  StringVector input_files;
  std::string model_file;
  bool eval_mode = false;
  bool running_text = false;
  bool force_tsv_output = false;
  std::string output_format;
  unsigned order = 1;

  parse_options(argc, argv, model_file, input_files, ner_config, order, running_text, eval_mode, output_format);

  if (running_text)
    ner_config.set_running_text_input(true);

  if (order > 3) {
    std::cerr << "crf-apply: Error: Currently, only the orders 1, 2 or 3 are supported" << std::endl;
    exit(2);
  }

  std::ifstream model_in(model_file.c_str(), std::ios::binary);
  if (!model_in) {
    std::cerr << "crf-apply: Error: Could not open binary model file '" << model_file << "'\n";
    exit(2);
  }

  if      (order == 1) 
    load_and_apply_model<1>(model_in, model_file, input_files, ner_config, eval_mode, output_format);
  else if (order == 2) 
    load_and_apply_model<2>(model_in, model_file, input_files, ner_config, eval_mode, output_format);
  else if (order == 3) 
    load_and_apply_model<3>(model_in, model_file, input_files, ner_config, eval_mode, output_format);
}


template<unsigned ORDER>
void load_and_apply_model(std::ifstream& model_in, const std::string& model_file, 
                          const StringVector& input_files, const NERConfiguration& ner_config, 
                          bool eval_mode, const std::string& output_format)
{
  std::cerr << "Loading model '" << model_file << "'\n";
  SimpleLinearCRFModel<ORDER> crf_model(model_in,true);
  model_info(crf_model);

  // Construct the applier
  CRFApplier<ORDER> crf_applier(crf_model,ner_config);

  /// Construct the outputter object
  NEROneWordPerLineOutputter one_word_per_line_outputter(std::cout);
  NERAnnotationOutputter ner_annotation_outputter(std::cout);
  JSONOutputter json_outputter(std::cout);
  NEROutputterBase* outputter = &one_word_per_line_outputter;
  if (output_format == "JSON") outputter = &json_outputter;
  //if (!force_tsv_output && ner_config.input_is_running_text()) 
  //  outputter = &json_outputter; // &ner_annotation_outputter;

  //std::cerr << "Reading in NE lists:";
  //load_clue_lists(crf_applier);
  //std::cerr << std::endl;

  for (unsigned i = 0; i < input_files.size(); ++i) {
    std::cerr << "Processing input file '" << input_files[i] << "'" << std::endl;
    std::ifstream test_data_in(input_files[i].c_str());
    if (!test_data_in) {
      std::cerr << "crf-apply: Error opening file '" << input_files[i] << "'" << std::endl;
      continue;
    }

    time_t t0 = clock();
    outputter->prolog();
    if (eval_mode) {
      EvaluationInfo e = crf_applier.evaluation_of(test_data_in,*outputter);
      std::cerr << "  Accuracy:  " << (e.accuracy()*100) << "%\n";
      std::cerr << "  Precision: " << (e.precision()*100) << "%\n";
      std::cerr << "  Recall:    " << (e.recall()*100) << "%\n";
    }
    else {
      crf_applier.apply_to(test_data_in,*outputter);
    }
    outputter->epilog();
    unsigned t = clock() - t0;

    std::cerr << "  Processed " << crf_applier.processed_tokens() << " tokens in " 
              << crf_applier.processed_sequences() << " sequences in " << t << "ms ";
    if (t > 0) 
      std::cerr << "(" << (1000* crf_applier.processed_tokens() / float(t)) << " tokens/s)\n";
    else 
      std::cerr << std::endl;
  } // for i
}


void parse_options(int argc, char* argv[], std::string& model_file, 
                   StringVector& input_files, NERConfiguration& ner_config, 
                   unsigned& order, bool& running_text, bool& eval_mode, 
                   std::string& output_format)
{
  typedef TCLAP::ValueArg<std::string>  StringValueArg;
  typedef TCLAP::SwitchArg              BoolArg;
  typedef TCLAP::ValueArg<unsigned>     IntValueArg;

  if (argc == 1) {
    usage();
  }

  try {
    TCLAP::CmdLine cmd("crf-apply -- Applies a trained CRF model to a input textfile\n",' ',"1.0");
    StringValueArg config_file_arg("c","config","Configuration file",true,"","filename");
    StringValueArg model_file_arg("m","model","Binary model file",true,"","filename");
    IntValueArg order_arg("o","order","Model order",true,1,"1,2 or 3");
    BoolArg running_text_arg("r","running-text","Running text (as opposed to tab-separated column style data)",false);
    BoolArg eval_mode_arg("e","eval","Puts crf-apply into evaluation mode",false);
    StringValueArg output_format_arg("f","format","Output format ",false,"TSV","TSV,JSON");
    TCLAP::UnlabeledMultiArg<std::string> input_files_arg("input","input files",true,"input-filename");

    cmd.add(model_file_arg);
    cmd.add(config_file_arg);
    cmd.add(input_files_arg);
    cmd.add(output_format_arg);
    cmd.add(eval_mode_arg);
    cmd.add(running_text_arg);
    cmd.add(order_arg);

    cmd.parse(argc,argv);

    model_file = model_file_arg.getValue();
    eval_mode = eval_mode_arg.getValue();
    if (output_format_arg.getValue() == "JSON" || output_format_arg.getValue() == "TSV") {
      output_format = output_format_arg.getValue();
    }
    else {
      std::cerr << "crf-apply" << ": Error: Invalid output format '" << output_format_arg.getValue() << "'" << std::endl;
    }
    running_text = running_text_arg.getValue();
    order = order_arg.getValue();

    std::string conf_file = config_file_arg.getValue();
    if (!conf_file.empty()) {
      std::ifstream conf_in(conf_file.c_str());
      if (conf_in) {
        std::cerr << "Loading configuration file '" << conf_file << "'" << std::endl;
        ner_config.read_config_file(conf_in);
        std::cerr << std::endl;
      }
      else {
        std::cerr << "crf-apply" << ": Error loading configuration file '" << conf_file << "'" << std::endl;
      }
    }

    input_files = input_files_arg.getValue();
  }

  catch (TCLAP::ArgException &e) { // catch any exceptions
    std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
  }
} 


void usage()
{
  std::cerr << "crf-apply" << " (order=" << MODEL_ORDER << ")" << std::endl;
  std::cerr << "Usage: " << "crf-apply" << " -c CONFIG-FILE -m MODEL-FILE [-e] TEXT-FILE ..." << std::endl << std::endl;
  std::cerr << "  CONFIG-FILE is the configuration file" << std::endl;
  std::cerr << "  MODEL-FILE is the binary file as produced by crf-train or crf-convert" << std::endl;
  std::cerr << "  TEXT-FILE is a standard ISO text file (UTF-8 will be supported soon)" << std::endl;
  std::cerr << "  -e puts crf-apply into evaluation mode (this assumes a special annotation in the input text files)\n";
  std::cerr << std::endl << "Example: crf-apply -c ner.cfg -m mymodel.crf" << std::endl;
  exit(1);
}

template<unsigned O> 
void load_clue_lists(CRFApplier<O>& crf_applier)
{
  const char* fn1 = NER_LIST_ALL;
  std::ifstream list_in1(fn1,std::ios::binary);
  if (list_in1) {
    std::cerr << " " << fn1;
    //crf_applier.add_ne_list(list_in1);
  }
  else std::cerr << "\nError: crf-test: Unable to open NE list '" << fn1 << "'" << std::endl;

  const char* fn2 = NE_LEFTCONTEXT_LIST;
  std::ifstream list_in2(fn2,std::ios::binary);
  if (list_in2) {
    std::cerr << " " << fn2;
    crf_applier.add_left_context_list(list_in2);
  }
  else std::cerr << "\nError: crf-test: Unable to open context list '" << fn2 << "'" << std::endl;

  const char* fn3 = NE_RIGHTCONTEXT_LIST;
  std::ifstream list_in3(fn3,std::ios::binary);
  if (list_in3) {
    std::cerr << " " << fn3;
    crf_applier.add_left_context_list(list_in3);
  }
  else std::cerr << "\nError: crf-test: Unable to open context list '" << fn3 << "'" << std::endl;

  const char* fn4 = PERSONNAMES_LIST;
  std::ifstream list_in4(fn4,std::ios::binary);
  if (list_in4) {
    std::cerr << " " << fn4;
    crf_applier.add_person_names_list(list_in4);
  }
  else std::cerr << "\nError: crf-test: Unable to open person name list '" << fn4 << "'" << std::endl;
}

