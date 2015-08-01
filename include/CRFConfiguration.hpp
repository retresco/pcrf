////////////////////////////////////////////////////////////////////////////////
// CRFConfiguration.hpp
// Defines a class which holds a CRF configuration
// TH, June 2015
// TODO:
//   * Windows CR/LF
//   * quiet mode
////////////////////////////////////////////////////////////////////////////////

#ifndef __CRF_CONFIGURATION_HPP__
#define __CRF_CONFIGURATION_HPP__

/**
  \page CRFConfiguration CRF configurations
  An instance of CRFConfiguration represents the current state the configuration
  which controls the annotation of input sequences (see CRFFeatureExtractor) and
  the application of a CRF model to test data (see CRFApplier).\n
  A CRF configuration consists out of <b>internal attributes</b> to add to the input 
  (that is, N-grams, contexts, capitalisation and normalisation, prefixes and 
  suffixes, and <b>external attributes</b>, that is attributes taken from external
  sources like lexicons, Wikipedia lists etc.
  
  The most convenient way to control annotation and application is to use a 
  <b>configuration file</b>. Such a file is a text file which constists of 
  key-value pairs and is read by the constructor of CRFConfiguration. 
  The table in the following section explains the available keys and their values
  in greater detail.
 
  \section CRFConfigurationKeysAndValues Format of a configuration file
  <table>
  <tr><th><b>Key</b></th><th><b>Explanation and possible value(s)</b></th></tr>
  <tr><td><tt>AnnotationScheme</tt></td>
  <td><tt>bilou</tt> or <tt>bio</tt>. Annotation schemes add a state mechanism to the output labels.
  <tt>B</tt> stands for the beginning of some annotated sequence, 
  <tt>I</tt> means: within a sequence,
  <tt>L</tt> means: last element of the sequence and 
  <tt>U</tt> stands for unit-length sequences.
  <tt>O</tt> stands for: outside a sequence</td></tr>
  <tr><td><tt>DefaultLabel</tt></td>
  <td>Determines the default label (any string)</td></tr>
  <tr><td><tt>OutputToken</tt></td>
  <td><tt>yes</tt> or <tt>no</tt>. Should the input token be also outputted?</td></tr>
  <tr><td><tt>RunningText</tt></td><td><tt>yes</tt> or <tt>no</tt>. 
  Is the input running text or table data?</td></tr>
  <tr><td><tt>Patterns</tt></td>
  <td>Name of a binary file which contains a <i>Directed Acyclic Word Graph</i>
  which contains known input sequences which should be annotated with
  specified features. Examples include Wikipedia features, lexicons, 
  multi word expressions etc.</td></tr>
  <tr><td><tt>LeftContextFilename</tt></td>
  <td>Same as <tt>Patterns</tt>, but for left contexts<td></td></tr>
  <tr><td><tt>RightContextFilename</tt></td>
  <td>Same as <tt>LeftContextFilename</tt>, but for right contexts</td></tr>
  <tr><td><tt>RegexFilename</tt></td>
  <td>A text file where each line consists of a regular expression (in boost regex syntax) </td></tr>
  </table>
  The following table shows the available, predefined attributes. 
  Their values are -- unless otherwise stated -- always <tt>yes</tt> or <tt>no</tt>.
  <table>
  <tr><th><b>Key</b></th><th><b>Explanation and possible value(s)</b></th></tr>
  <tr><td><tt>AllPatterns</tt></td>
  <td></td></tr>
  <tr><td><tt>HeadWord</tt></td>
  <td></td></tr>
  <tr><td><tt>HeadWordLowercased</tt></td>
  <td></td></tr>
  <tr><td><tt>AllPrevWords</tt></td>
  <td></td></tr>
  <tr><td><tt>AllNextWords</tt></td>
  <td></td></tr>
  <tr><td><tt>TokenClass</tt></td>
  <td></td></tr>
  <tr><td><tt></tt></td>
  <td></td></tr>
  </table>

  \section DAWGInputfiles Format of DAWG input files
  External information like lists of sequences which are treated specially during
  the annotation process is efficiently stored in the form of a 
  <i>Directed Acyclic Word Graph</i>, that is a minimal and deterministic automaton.
  These DAWGs are created form input files with at least two columns where the first
  one contains a specified attribute value V and the remaining columns contain the sequence
  which receives that value if found as a subsequence of an input sequences.
  If found, an attribute value of the form <tt>Pattern[start..end]=V</tt> is instantiated
  to the elements this subsequence.
  
*/

#include <string>
#include <iostream>
#include <fstream>
#include <map>

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

#include "CRFFeatureExtractor.hpp"

/// CRFConfiguration
class CRFConfiguration
{
private:
  typedef boost::char_separator<char>     CharSeparator;
  typedef boost::tokenizer<CharSeparator> Tokenizer;

public:
  /// Empty configuration
  CRFConfiguration() 
  {
    init();
  }

  /// Read configuration from text stream
  CRFConfiguration(std::istream& config_in) 
  {
    init();
    read_config_file(config_in);
  }

  /// Read configuration from text file named 'config_file'
  CRFConfiguration(std::string config_file) 
  {
    init();
    std::ifstream config_in(config_file.c_str());
    if (config_in)
      read_config_file(config_in);
  }

  /// Read the configuration from an input stream
  void read_config_file(std::istream& config_in)
  {
    const CharSeparator segmenter("\t ","=");
    std::string line;
    std::vector<std::string> tokens;

    while (config_in.good()) {
      std::getline(config_in,line);
      if (line.empty() || line[0] == '#') continue;
      Tokenizer tokenizer(line,segmenter);
      tokens.assign(tokenizer.begin(),tokenizer.end());
      if (tokens.size() == 3 && tokens[1] == "=") {
        if (tokens[0] == "Columns")                                         set_columns(tokens[2]);
        else if (tokens[0] == "DefaultLabel") {
          default_label = tokens[2];
          //std::cerr << "  DefaultLabel      = " << default_label << std::endl;
        }
        
        else if (tokens[0] == "Patterns")                                   patterns_list_filename = tokens[2];
        //else if (tokens[0] == "PersonsFilename")                            personnames_list_filename = tokens[2];
        else if (tokens[0] == "LeftContextFilename")                        right_context_filename = tokens[2];
        else if (tokens[0] == "RightContextFilename")                       left_context_filename = tokens[2];
        else if (tokens[0] == "RegexFilename")                              regex_filename = tokens[2];
        
        else if (tokens[0] == "OutputToken") {
          output_tok = bool_value(tokens[2]);
          std::cerr << "  OutputToken       = " << (output_tok ? "yes":"no") << std::endl;
        }
        
        else if (tokens[0] == "RunningText") {
          running_text_input = bool_value(tokens[2]);
          std::cerr << "  RunningText       = " << (running_text_input ? "yes":"no") << std::endl;
        }
        
        else if (tokens[0] == "InnerWordNgrams" && bool_value(tokens[2])) {
          inner_word_ngrams = bool_value(tokens[2]);
          std::cerr << "  InnerWordNgrams   = " << (inner_word_ngrams ? "yes":"no") << std::endl;
        }
        
        else if (tokens[0] == "ModelOrder") {
          //unsigned o = boost::lexical_cast<unsigned>(tokens[2]);
          //if (o == 1 || o == 2) {
          //  order = o;
          //}
          //else std::cerr << "NERConfiguration: model order is restricted to 1 or 2" << std::endl;
        }
        
        else if (tokens[0] == "AnnotationScheme") {
          if      (tokens[2] == "bio")    anno_scheme = nerBIO;
          else if (tokens[2] == "bilou")  anno_scheme = nerBILOU;
          else std::cerr << "NERConfiguration: AnnotationScheme must be either 'bio' or 'bilou'" << std::endl;
          std::cerr << "  Annotation        = " << scheme_to_string(anno_scheme) << std::endl;
        }
        
        else if (tokens[0] == "ContextWindowSize") {
          set_context_window_size(boost::lexical_cast<unsigned>(tokens[2]));
          std::cerr << "  ContextWindowSize = " << context_window_size << std::endl;
        }
        
        else if (tokens[0] == "NGramWindowSize") {
          set_ngram_window_size(boost::lexical_cast<unsigned>(tokens[2]));
          std::cerr << "  NGramWindowSize = " << ngram_window_size << std::endl;
        }

        else if (bool_value(tokens[2])) {
          // Assume that everything else is a feature
          std::cerr << "  Use feature       : " << tokens[0] << std::endl;
          add_feat(tokens[0]);
        }
      }
    } // while
  }

  /// Return the currently selected features as a bit vector
  FeatureType features()                        const { return feats; }
  /// 
  bool output_token()                           const { return output_tok; }
  /// Is the input running text or table data
  bool input_is_running_text()                  const { return running_text_input; }
  unsigned model_order()                        const { return order; }
  NERAnnotationScheme annotation_scheme()       const { return anno_scheme; }
  void set_output_token(bool v)                 { output_tok = v; }
  void set_running_text_input(bool v)           { running_text_input = v; }
  void set_model_order(unsigned o)              { order = o; }
  unsigned columns_count()                      const { return columns.size(); }
  unsigned get_context_window_size()            const { return context_window_size; }
  void set_context_window_size(unsigned n)      { if (n > 0) context_window_size = n; }
  unsigned get_ngram_window_size()              const { return ngram_window_size; }
  void set_ngram_window_size(unsigned n)        { if (n > 1) ngram_window_size = n; }
  std::string get_default_label()               const { return default_label; }
  void set_default_label(const std::string& l)  { default_label = l; }
  bool get_inner_word_ngrams()                  const { return inner_word_ngrams; }
  void set_inner_word_ngrams(bool v)            { inner_word_ngrams = v; }

  unsigned get_column_no(const std::string& name) const 
  {
    auto f = columns.find(name);
    return (f != columns.end()) ? f->second : unsigned(-1);
  }

  std::string get_filename(const std::string& key) const
  {
    if (key == "Patterns") return patterns_list_filename;
    return "";
  }

  unsigned get_max_prefix_length()            const { }
  void set_max_prefix_length(unsigned l)      { }

  unsigned get_max_suffix_length()            const { }
  void set_max_suffix_length(unsigned l)      { }

  std::string get_patterns_list_filename()    const { return patterns_list_filename; } 
  //std::string get_personnames_list_filename() const { return personnames_list_filename; } 
  std::string get_right_context_filename()    const { return right_context_filename; } 
  std::string get_left_context_filename()     const { return left_context_filename; } 
  std::string get_regex_filename()            const { return regex_filename; } 

  void add_feat(FeatureType f)                { feats |= f; }
  void add_feat(const std::string& f)         { feats |= translate(f); }

  void add_feats(const std::string& f_string)
  {    
    Tokenizer tokenizer(f_string,CharSeparator("+|,; "));
    for (Tokenizer::const_iterator f = tokenizer.begin(); f != tokenizer.end(); ++f) {
      add_feat(*f);
    }
  }

  void reset()
  {
    feats = 0;
    output_tok = false;
    order = 1;
  }

private:
  void init()
  {
    feats = 0;
    anno_scheme = nerBIO; 
    default_label = "OTHER"; 
    output_tok = false; 
    running_text_input = false; 
    order = 1;
    context_window_size = 4;
    ngram_window_size = 2;
    max_word_prefix_length = 4;
    max_word_suffix_length = 4;
    inner_word_ngrams = false;
  }

  FeatureType translate(const std::string& feat) const
  {
    FeatureType f = 0;
    if      (feat == "HeadWord")              f = HeadWord;
    else if (feat == "HeadWordLowercased")    f = HeadWordLowercased;
    else if (feat == "AllWords")              f = AllWords;
    else if (feat == "AllPrevWords")          f = AllPrevWords;
    else if (feat == "AllNextWords")          f = AllNextWords;
    else if (feat == "AllPrefixes")           f = AllPrefixes;
    else if (feat == "AllSuffixes")           f = AllSuffixes;
    else if (feat == "AllPosTags")            f = AllPosTags;
    else if (feat == "AllLemmas")             f = AllLemmas;
    else if (feat == "AllDelim")              f = AllDelim;
    else if (feat == "AllWBigrams")           f = AllW2grams;
    else if (feat == "AllWTrigrams")          f = AllW3grams;
    else if (feat == "AllWTetragrams")        f = AllW4grams;
    else if (feat == "AllWPentagrams")        f = AllW5grams;
    else if (feat == "AllWHexagrams")         f = AllW6grams;
    else if (feat == "AllWHeptagrams")        f = AllW7grams;
    else if (feat == "AllWOctagrams")         f = AllW8grams;
    else if (feat == "AllWNonagrams")         f = AllW9grams;
    else if (feat == "AllWDecagrams")         f = AllW10grams;
    else if (feat == "AllWNgrams")            f = AllWNgrams;
    else if (feat == "AllPOSBigrams")         f = AllT2grams;
    else if (feat == "AllPOSTrigrams")        f = AllT3grams;
    else if (feat == "AllPOSgrams")           f = AllTNgrams;
    else if (feat == "AllTokenTypes")         f = AllTokenTypes;
    //else if (feat == "AllPersonNames")        f = AllPersonNames;
    //else if (feat == "AllNamedEntities")      f = AllNamedEntities;
    else if (feat == "AllListFeatures")       f = AllListFeatures;
    else if (feat == "AllPatterns")           f = AllPatterns;
    else if (feat == "AllContextClues")       f = AllContextClues;
    else if (feat == "AllListFeatures")       f = AllListFeatures;
    else if (feat == "AllRegexes")            f = AllRegexes;
    else if (feat == "AllCharNgrams")         f = AllCharNgrams;
    else if (feat == "LeftContextContains")   f = LeftContextContains;
    else if (feat == "RightContextContains")  f = RightContextContains;
    else if (feat == "AllContextContains")    f = AllContextContains;
    else if (feat == "AllInitUpper2grams")    f = AllInitUpper2grams;
    else if (feat == "AllInitUpper3grams")    f = AllInitUpper3grams;
    else if (feat == "AllInitUpperGrams")     f = AllInitUpperGrams;
    else if (feat == "AllShapes")             f = AllShapes;
    else if (feat == "WordPOS")               f = WordPOS;
    else if (feat == "TokenClass")            f = TokenClass;
    else if (feat == "VCPattern")             f = VCPattern;
    else std::cerr << "  Error: Unknown feature group '" << feat << "'" << std::endl;
    return f;
  }

  bool bool_value(const std::string& v) const
  {
    if (v == "yes" || v == "true" || v == "1") return true;
    if (v == "no" || v == "false" || v == "0") return false;
    std::cerr << "CRFConfiguration: invalid value '" << v << "'" << std::endl;
    return true;
  }

  std::string scheme_to_string(NERAnnotationScheme anno_scheme) const
  {
    if (anno_scheme == nerBILOU)  return "BILOU";
    if (anno_scheme == nerBIO)    return "BIO";
    return "";
  }

  void set_columns(const std::string& col_str)
  {
    const CharSeparator segmenter(";|");
    Tokenizer tokenizer(col_str,segmenter);
    unsigned i = 0;
    for (auto t = tokenizer.begin(); t != tokenizer.end(); ++t, ++i) {
      columns[*t] = i;
    }
  }

private:
  typedef std::map<std::string,unsigned> ColumnsMap;

private:
  FeatureType         feats;
  NERAnnotationScheme anno_scheme; 
  std::string         default_label;
  std::string         patterns_list_filename;
  //std::string         personnames_list_filename;
  std::string         right_context_filename;
  std::string         left_context_filename;
  std::string         regex_filename;
  ColumnsMap          columns;
  bool                output_tok;
  bool                running_text_input;
  bool                inner_word_ngrams;
  unsigned            order;
  unsigned            context_window_size;
  unsigned            ngram_window_size;
  unsigned            max_word_prefix_length;
  unsigned            max_word_suffix_length;
}; // CRFConfiguration

#endif
