////////////////////////////////////////////////////////////////////////////////////////////////////
// crf-apply.cpp
// Application and evaluation of CRF models
// TH, Juli 2015
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
  \page CRFCommands Commands-line tools
  \section CRFApply crf-apply
*/

#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <limits>

#include <tclap/CmdLine.h>

#include "../include/CRFApplier.hpp"
#include "../include/SimpleLinearCRFModel.hpp"
#include "../include/CRFDecoder.hpp"
#include "../include/CRFUtils.hpp"
#include "../include/CRFFeatureExtractor.hpp"
#include "../include/CRFConfiguration.hpp"
#include "../include/AsyncTokenizer.hpp"
#include "../include/NEROutputters.hpp"
#include "../include/TokenWithTag.hpp"


// TODO: integrate this into the NERConfig
#define PERSONNAMES_LIST        "lists/personnames.list.bin"
#define NER_LIST_ALL            "lists/named-entities.list.bin"
#define NE_LEFTCONTEXT_LIST     "lists/lc.list.bin"
#define NE_RIGHTCONTEXT_LIST    "lists/rc.list.bin"
#define NE_REGEX_LIST           "lists/regex.list"

#define PROGNAME                "crf-apply"

typedef std::vector<std::string>   StringVector;

// Prototypes
void parse_options(int argc, char* argv[], std::string&, StringVector&, CRFConfiguration&, unsigned&, bool&, bool&, std::string&);
template<unsigned O> void load_and_apply_model(std::ifstream&,const std::string&,const StringVector&,const CRFConfiguration&, bool, bool, const std::string&);
void show_evaluation_results(const EvaluationInfo&, const LabelSet&);
template<unsigned O> void load_clue_lists(CRFApplier<O>&);
void usage();
void banner();

/// main function
int main(int argc, char* argv[])
{
  CRFConfiguration crf_config;
  StringVector input_files;
  std::string model_file;
  bool eval_mode = false;
  bool running_text = false;
  bool force_tsv_output = false;
  std::string output_format;
  unsigned order = 1;

  banner();
  parse_options(argc, argv, model_file, input_files, crf_config, order, running_text, eval_mode, output_format);

//  if (running_text)
//    ner_config.set_running_text_input(true);

  if (order > 3) {
    std::cerr << PROGNAME << ": Error: Currently, only the orders 1, 2 or 3 are supported" << std::endl;
    exit(2);
  }

  std::ifstream model_in(model_file.c_str(), std::ios::binary);
  if (!model_in) {
    std::cerr << PROGNAME << ": Error: Could not open binary model file '" << model_file << "'\n";
    exit(2);
  }

  if (order == 1) 
    load_and_apply_model<1>(model_in, model_file, input_files, crf_config, running_text, eval_mode, output_format);
  else if (order == 2) 
    load_and_apply_model<2>(model_in, model_file, input_files, crf_config, running_text, eval_mode, output_format);
  else if (order == 3) 
    load_and_apply_model<3>(model_in, model_file, input_files, crf_config, running_text, eval_mode, output_format);
}


template<unsigned ORDER>
void load_and_apply_model(std::ifstream& model_in, const std::string& model_file, 
                          const StringVector& input_files, const CRFConfiguration& crf_config, 
                          bool running_text, bool eval_mode, const std::string& output_format)
{
  std::cerr << "Loading model '" << model_file << "'\n";
  SimpleLinearCRFModel<ORDER> crf_model(model_in,true);
  model_info(crf_model);

  // Construct the applier
  CRFApplier<ORDER> crf_applier(crf_model,crf_config);

  /// Construct the outputter object
  NEROneWordPerLineOutputter one_word_per_line_outputter(std::cout);
  NERAnnotationOutputter ner_annotation_outputter(std::cout);
  JSONOutputter json_outputter(std::cout);
  MorphOutputter morph_outputter(std::cout);
  NEROutputterBase* outputter = &one_word_per_line_outputter;
  if (output_format == "json") outputter = &json_outputter;
  else if (output_format == "single-line") outputter = &morph_outputter;

  //std::cerr << "Reading in NE lists:";
  //load_clue_lists(crf_applier);
  //std::cerr << std::endl;

  for (unsigned i = 0; i < input_files.size(); ++i) {
    std::cerr << "Processing input file '" << input_files[i] << "'" << std::endl;
    std::ifstream test_data_in(input_files[i].c_str());
    if (!test_data_in) {
      std::cerr << PROGNAME << ": Error opening file '" << input_files[i] << "'" << std::endl;
      continue;
    }

    time_t t0 = clock();
    outputter->prolog();
    if (eval_mode) {
      EvaluationInfo e = crf_applier.evaluation_of(test_data_in,*outputter,running_text);
      show_evaluation_results(e, crf_model.get_labels());
    }
    else {
      crf_applier.apply_to(test_data_in,*outputter,running_text);
    }
    outputter->epilog();
    time_t t = float(clock() - t0) * 1000 / CLOCKS_PER_SEC;

    std::cerr << "Processed " << crf_applier.processed_tokens() << " tokens in " 
              << crf_applier.processed_sequences() << " sequences in " << (t/1000) << "s ";
    if (t > 0) 
      std::cerr << "(" << (1000* crf_applier.processed_tokens() / float(t)) << " tokens/s)\n";
    else 
      std::cerr << std::endl;
  } // for i
}

/// Output evaluation statistics
void show_evaluation_results(const EvaluationInfo& e, const LabelSet& labels) 
{
  const std::string equals(50,'=');
  const std::string dashes(50,'-');

  std::cerr << std::endl << equals << std::endl;
  std::cerr << "Evaluation\n";
  std::cerr << equals << std::endl;
  std::cerr << "Global accuracy:    " << e.accuracy() << "\n";
  //std::cerr << "Averaged precision: " << e.precision() << "\n";
  //std::cerr << "Averaged recall:    " << e.recall() << "\n";
  std::cerr << "\nPer label precision/recall/F1-score:";
  std::cerr << std::endl << dashes << std::endl;
  std::cerr << "Label                   Prec      Rec       F1\n";
  std::cerr << dashes << std::endl;
  for (auto l = labels.begin(); l != labels.end(); ++l) {
    float prec = e.precision(*l);
    if (prec > 0.0) {
      std::cerr << std::left << std::setw(20) << *l; 
      std::cerr << std::right << std::setw(10) << std::setprecision(4) << (prec);
      std::cerr << std::right << "    " << std::setprecision(4) << (e.recall(*l));
      std::cerr << std::endl;
    }
  }
  std::cerr << dashes << std::endl;
}

void parse_options(int argc, char* argv[], std::string& model_file, 
                   StringVector& input_files, CRFConfiguration& crf_config, 
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
    IntValueArg order_arg("o","order","Model order",false,1,"1,2 or 3");
    BoolArg running_text_arg("r","running-text","Running text (as opposed to tab-separated column style data)",false);
    BoolArg eval_mode_arg("e","eval","Puts crf-apply into evaluation mode",false);
    StringValueArg output_format_arg("f","format","Output format ",false,"tsv","tsv,json,single-line");
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

    std::set<std::string> output_formats;
    output_formats.insert("tsv"); output_formats.insert("json"); output_formats.insert("single-line");
    if (output_formats.find(output_format_arg.getValue()) != output_formats.end()) {
      output_format = output_format_arg.getValue();
    }
    else {
      std::cerr << PROGNAME << ": Error: Invalid output format '" << output_format_arg.getValue() << "'" << std::endl;
    }
    
    running_text = running_text_arg.getValue();
    order = order_arg.getValue();

    std::string conf_file = config_file_arg.getValue();
    if (!conf_file.empty()) {
      std::ifstream conf_in(conf_file.c_str());
      if (conf_in) {
        std::cerr << "Loading configuration file '" << conf_file << "'" << std::endl;
        crf_config.read_config_file(conf_in);
        std::cerr << std::endl;
      }
      else {
        std::cerr << PROGNAME << ": Error loading configuration file '" << conf_file << "'" << std::endl;
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
  std::cerr << "Usage: " << "crf-apply" << " -c CONFIG-FILE -m MODEL-FILE [-e] [-r] [-f OUTPUT-TYPE] TEXT-FILE ..." << std::endl << std::endl;
  std::cerr << "  CONFIG-FILE is the configuration file" << std::endl;
  std::cerr << "  MODEL-FILE is the binary file as produced by crf-train or crf-convert" << std::endl;
  std::cerr << "  TEXT-FILE is a standard UTF-8-encoded text file" << std::endl;
  std::cerr << "  OUTPUT-TYPE determines the form of the output: 'tsv' means column-style, 'json' is JSON-output\n";
  std::cerr << "  -e puts crf-apply into evaluation mode (this assumes a special annotation in the input text files)\n";
  std::cerr << "  -r tells crf-apply to assume a running text file (as opposed to a tab-separated input file)\n";
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
  else std::cerr << "\n" << PROGNAME << ": Error: Unable to open NE list '" << fn1 << "'" << std::endl;

  const char* fn2 = NE_LEFTCONTEXT_LIST;
  std::ifstream list_in2(fn2,std::ios::binary);
  if (list_in2) {
    std::cerr << " " << fn2;
    crf_applier.add_left_context_list(list_in2);
  }
  else std::cerr << "\n" << PROGNAME << ": Error: Unable to open context list '" << fn1 << "'" << std::endl;

  const char* fn3 = NE_RIGHTCONTEXT_LIST;
  std::ifstream list_in3(fn3,std::ios::binary);
  if (list_in3) {
    std::cerr << " " << fn3;
    crf_applier.add_left_context_list(list_in3);
  }
  else std::cerr << "\n" << PROGNAME << ": Error: Unable to open context list '" << fn1 << "'" << std::endl;

  const char* fn4 = PERSONNAMES_LIST;
  std::ifstream list_in4(fn4,std::ios::binary);
  if (list_in4) {
    std::cerr << " " << fn4;
    crf_applier.add_person_names_list(list_in4);
  }
  else std::cerr << "\n" << PROGNAME << ": Error: Unable to NE list '" << fn1 << "'" << std::endl;
}

void banner()
{
  std::cerr << PROGNAME << " (";
  #ifdef PCRF_UTF8_SUPPORT
    std::cerr << "UTF-8 encoding";
  #else
    std::cerr << "Latin1 encoding";
  #endif
  std::cerr << ")" << std::endl;
}

