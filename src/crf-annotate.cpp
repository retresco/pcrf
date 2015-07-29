////////////////////////////////////////////////////////////////////////////////////////////////////
// crf-annotate.cpp
// TH, March 2015
// Reads a test corpus file (either running text or tab-separated) 
// and creates the input file for the CRF training phase
// TODO:
//  * Generalise input formats
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
  \page CRFCommands Commands-line tools
  \section CRFAnnotate crf-annotate
  <tt>crf-annotate</tt>\
  Synopsis:
*/

#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iterator>
#include <algorithm>

#include <boost/tokenizer.hpp>
#include <tclap/CmdLine.h>

//#define USE_BOOST_REGEX
#include "../include/CRFFeatureExtractor.hpp"
#include "../include/CRFConfiguration.hpp"
#include "../include/AsyncTokenizer.hpp"
#include "../include/TokenWithTag.hpp"
#include "../include/CRFHelpers.hpp"

#define PROGNAME                "crf-annotate"

// Typedefs
typedef std::vector<std::string>   StringVector;

// Prototypes
void parse_options(int, char**, StringVector&, CRFConfiguration&, bool&);
unsigned process_text(std::ifstream&, CRFConfiguration&, const CRFFeatureExtractor&);
unsigned process_column_data(std::ifstream&, CRFConfiguration&, const CRFFeatureExtractor&);
void usage();
void banner();


int main(int argc, char* argv[])
{
  StringVector input_files;
  CRFConfiguration crf_config;
  unsigned n_seq;
  bool running_text = false;

  banner();
  
  parse_options(argc, argv, input_files, crf_config, running_text);
  
  WordWithAttributes::SetOutputTokenFlag(crf_config.output_token());

  // Initialise the feature extractor
  CRFFeatureExtractor crf_fe(crf_config.features(),false);
  // Set the window size for InLC/InRC features
  crf_fe.set_context_window_size(crf_config.get_context_window_size());
  // Check for tag column
  crf_fe.have_pos_tags(crf_config.get_column_no("Tag") != unsigned(-1));
  // Inner word N-grams
  crf_fe.set_inner_word_ngrams(crf_config.get_inner_word_ngrams());

  std::cerr << "Reading lists:";
  if ((crf_config.features() & AllPatterns) != 0) {
    std::string fn = crf_config.get_filename("Patterns");
    if (fn.empty()) {
      std::cerr << "Warning: 'AllPatterns' specified, but no filename for 'Patterns' key given\n";
    }
    else {
      load_binary_ne_list(fn,crf_fe);
    }
  }

  //if ((crf_config.features() & AllPersonNames) != 0) {
  //  load_binary_person_names_list(PERSONNAMES_LIST,crf_fe);
  //}

  //if ((crf_config.features() & AllContextClues) != 0) {
  //  load_binary_context_list(NE_LEFTCONTEXT_LIST,crf_fe,true);
  //  load_binary_context_list(NE_RIGHTCONTEXT_LIST,crf_fe,false);
  //}

  //if ((crf_config.features() & AllRegexes) != 0) {
  //  load_regex_list(NE_REGEX_LIST,crf_fe);
  //}
  std::cerr << std::endl;

  for (unsigned i = 0; i < input_files.size(); ++i) {
    std::ifstream data_in(input_files[i].c_str());
    if (!data_in) {
      std::cerr << "Error: " << PROGNAME << " invalid training data file '" 
                << input_files[i] << "' " << std::endl;
      exit(2);
    }

    std::cerr << "Processing '" << input_files[i] << "' ";
    time_t t0 = clock();
    if (crf_config.input_is_running_text() || running_text) {
      n_seq = process_text(data_in,crf_config,crf_fe);
    }
    else {
      n_seq = process_column_data(data_in,crf_config,crf_fe);
    }
    std::cerr << " done (" << (float(clock()-t0)*1000/CLOCKS_PER_SEC) << "ms)" << std::endl;
    std::cerr << "[" << n_seq << " sequences]\n";
  } // for i
}


unsigned process_text(std::ifstream& data_in, CRFConfiguration& crf_config, const CRFFeatureExtractor& crf_fe)
{
  TokenWithTagSequence sentence;
  AsyncTokenizer tokenizer(data_in,crf_config.annotation_scheme()==nerBILOU, 
                           crf_config.model_order(),crf_config.get_default_label());
  unsigned n_seq = 0;
  while (tokenizer.tokenize(sentence)) {
    if (sentence.size() > 0) {
      //std::cout << "Sentence # " << ++n << std::endl;
      //std::copy(sentence.begin(),sentence.end(),std::ostream_iterator<TokenWithTag>(std::cout,"\n"));
      ++n_seq;
      CRFInputSequence x = crf_fe.add_features(sentence);
      std::copy(x.begin(),x.end(),std::ostream_iterator<WordWithAttributes>(std::cout,"\n"));
      std::cout << std::endl;
      sentence.clear();
      if ((n_seq % 1000) == 0) std::cerr << ".";
    }
  } // while
  return n_seq;
}

unsigned process_column_data(std::ifstream& data_in, 
                             CRFConfiguration& crf_config, 
                             const CRFFeatureExtractor& crf_fe)
{
  std::vector<std::string> tokens;
  std::string line;
  TokenWithTagSequence sequence;
  unsigned n_seq = 0;
  unsigned col_count = crf_config.columns_count();
  unsigned token_column = crf_config.get_column_no("Token");
  unsigned label_column = crf_config.get_column_no("Label");
  unsigned tag_column = crf_config.get_column_no("Tag");
  unsigned position_column = crf_config.get_column_no("Position");
  unsigned lemma_column = crf_config.get_column_no("Lemma");

  if (token_column == unsigned(-1) || label_column == unsigned(-1)) {
    std::cerr << "Missing token/label column\n";
    return 0;
  }

  while (data_in.good()) {
    std::getline(data_in,line);
    if (line.empty()) {
      if (!sequence.empty()) {
        CRFInputSequence x = crf_fe.add_features(sequence);
        std::copy(x.begin(),x.end(),std::ostream_iterator<WordWithAttributes>(std::cout,"\n"));
        std::cout << std::endl;
        sequence.clear();
        if ((n_seq % 1000) == 0) std::cerr << ".";
        ++n_seq;
      }
    }
    else {
      boost::tokenizer<boost::char_separator<char>  > tokenizer(line, boost::char_separator<char>("\t "));
      tokens.assign(tokenizer.begin(),tokenizer.end());
      if (tokens.size() == col_count) {
        TokenWithTag tt(tokens[token_column]);
        tt.assign_label(tokens[label_column]);
        if (tag_column != unsigned(-1)) 
          tt.assign_tag(tokens[tag_column]);
        sequence.push_back(tt);
      }
    }
  } // while
  // TODO: process last sequence
  return n_seq;
}


void parse_options(int argc, char* argv[], StringVector& input_files, CRFConfiguration& crf_config, bool& running_text)
{
  typedef TCLAP::ValueArg<std::string>  StringValueArg;
  typedef TCLAP::SwitchArg              BoolArg;

  if (argc == 1) {
    usage();
  }

  try {
    TCLAP::CmdLine cmd("ner-annotate -- Annotates (+- annotated) UTF-8 texts for CRF-training\n",' ',"1.0");
    StringValueArg gen_feat_arg("f","feat","Features to be generated",false,"","\"Feat1|Feat2 ...\"");
    StringValueArg config_file_arg("c","config","configuration file",false,"","filename");
    BoolArg output_token_arg("t","output-token","Output token",false);
    BoolArg running_text_arg("r","running-text","Running text (as opposed to tab-separated column style data)",false);
    TCLAP::UnlabeledMultiArg<std::string> input_files_arg("input","input files",true,"input-filename");
    TCLAP::ValueArg<unsigned> order_arg("o","order","Markov order",false,1,"ORDER");

    cmd.add(order_arg);
    cmd.add(running_text_arg);
    cmd.add(output_token_arg);
    cmd.add(gen_feat_arg);
    cmd.add(config_file_arg);
    cmd.add(input_files_arg);

    cmd.parse(argc,argv);

    crf_config.set_output_token(output_token_arg.getValue());
    running_text = running_text_arg.getValue();

    unsigned order = order_arg.getValue();
    if (order < 1 || order > 2) {
      std::cerr << PROGNAME << ": Error" << ": " << "Only Markov orders of 1 (default) or 2 are permitted.\n";
    }
    else crf_config.set_model_order(order);

    std::string gen_feat_string = gen_feat_arg.getValue();
    if (!gen_feat_string.empty()) {
      crf_config.add_feats(gen_feat_string);
    }

    std::string conf_file = config_file_arg.getValue();
    if (!conf_file.empty()) {
      std::ifstream conf_in(conf_file.c_str());
      if (conf_in) {
        std::cerr << "Loading configuration file '" << conf_file << "'" << std::endl;
        crf_config.read_config_file(conf_in);
        std::cerr << std::endl;
      }
      else {
        std::cerr << PROGNAME << ": Error: Unable to open configuration file '" << conf_file << "'" << std::endl;
      }
    }

    input_files = input_files_arg.getValue();
  }

  catch (TCLAP::ArgException &e) { // catch any exceptions
    std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
  }
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

void usage()
{
  std::cerr << "Usage: " << PROGNAME << " [-c CONFIG-FILE] [-f \"FEAT-GEN-FLAGS\"] [-r] [-t] TRAINING-DATA" << std::endl;
  std::cerr << "  Annotated results are written to standard out" << std::endl;
  std::cerr << "  CONFIG-FILE is the configuration file" << std::endl;
  //std::cerr << "  ORDER is the Markov order of the model (1 or 2, default is 1)" << std::endl;
  std::cerr << "  FEAT-GEN-FLAGS = FEAT-GEN-GROUP [|FEAT-GEN-GROUP]*" << std::endl;
  std::cerr << "  FEAT-GEN-GROUP in { HeadWord|HeadWordLowercased|AllWords|AllWBigrams|AllWTrigrams|AllWNgrams|" << std::endl
            << "                      AllPrefixes|AllSuffixes|AllTokenTypes|AllShapes|TokenClass|AllCharNgrams|VCPattern|" << std::endl
            << "                      AllPrevWords|AllNextWords|" << std::endl
            << "                      AllPOSBigrams|AllPOSTrigrams|AllPosTags|AllPOSNgrams|WordTag|" << std::endl
            << "                      AllLemmas|AllDelim|AllContextContains|AllRegexes|" << std::endl
            << "                      AllPersonNames|AllNamedEntities|AllNELists|AllContextClues|AllListFeatures }" << std::endl;
  std::cerr << "  -t = output token\n";
  std::cerr << "  -r = running text (default is tab-separated column style data)\n";
  std::cerr << "Example:  ner-annotate test.txt.utf8 -f \"AllWords|AllPosTags\"" << std::endl;
  exit(1);
}
