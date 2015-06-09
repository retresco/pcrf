////////////////////////////////////////////////////////////////////////////////
// NERConfiguration.hpp
// TH, June 2015
// TODO:
//   * Windows CR/LF
//   * quiet mode
////////////////////////////////////////////////////////////////////////////////

#ifndef __NERCONFIGURATION_HPP__
#define __NERCONFIGURATION_HPP__

#include <string>
#include <iostream>
#include <fstream>
#include <map>

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

#include "NERFeatureExtractor.hpp"

/// NERConfiguration
class NERConfiguration
{
private:
  typedef boost::char_separator<char>     CharSeparator;
  typedef boost::tokenizer<CharSeparator> Tokenizer;

public:
  /// Empty configuration
  NERConfiguration() 
  : feats(0), anno_scheme(nerBIO), default_label("OTHER"), output_tok(false), 
    running_text_input(false), order(1), context_window_size(4) {}

  /// Read configuration from text stream
  NERConfiguration(std::istream& config_in) 
  : feats(0), anno_scheme(nerBIO), output_tok(false), order(1)
  {
    read_config_file(config_in);
  }

  /// Read configuration from text file
  NERConfiguration(std::string config_file) 
  : feats(0), anno_scheme(nerBIO), output_tok(false), order(1)
  {
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
          std::cerr << "  DefaultLabel     = " << default_label << std::endl;
        }
        
        else if (tokens[0] == "NamedEntities")                              ne_list_filename = tokens[2];
        else if (tokens[0] == "PersonsFilename")                            personnames_list_filename = tokens[2];
        else if (tokens[0] == "LeftContextFilename")                        right_context_filename = tokens[2];
        else if (tokens[0] == "RightContextFilename")                       left_context_filename = tokens[2];
        else if (tokens[0] == "RegexFilename")                              regex_filename = tokens[2];
        
        else if (tokens[0] == "OutputToken" && true_bool_value(tokens[2])) {
          output_tok = true;
          std::cerr << "  OutputToken       = yes" << std::endl;
        }
        
        else if (tokens[0] == "RunningText" && true_bool_value(tokens[2])) {
          running_text_input = true;
          std::cerr << "  RunningText      = yes" << std::endl;
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
        else if (true_bool_value(tokens[2])) {
          // Assume that everything else is a feature
          std::cerr << "  Use feature       : " << tokens[0] << std::endl;
          add_feat(tokens[0]);
        }
      }
    } // while
  }

  FeatureType features()                        const { return feats; }
  bool output_token()                           const { return output_tok; }
  bool input_is_running_text()                  const { return running_text_input; }
  unsigned model_order()                        const { return order; }
  NERAnnotationScheme annotation_scheme()       const { return anno_scheme; }
  void set_output_token(bool v)                 { output_tok = v; }
  void set_running_text_input(bool v)           { running_text_input = v; }
  void set_model_order(unsigned o)              { order = o; }
  unsigned columns_count()                      const { return columns.size(); }
  unsigned get_context_window_size()            const { return context_window_size; }
  void set_context_window_size(unsigned n)      { if (n > 0) context_window_size = n; }
  std::string get_default_label()               const { return default_label; }
  void set_default_label(const std::string& l)  { default_label = l; }

  unsigned get_column_no(const std::string& name) const 
  {
    auto f = columns.find(name);
    return (f != columns.end()) ? f->second : unsigned(-1);
  }

  std::string get_filename(const std::string& key) const
  {
    if (key == "NamedEntities") return ne_list_filename;
    return "";
  }

  unsigned get_max_prefix_length()            const { }
  void set_max_prefix_length(unsigned l)      { }

  unsigned get_max_suffix_length()            const { }
  void set_max_suffix_length(unsigned l)      { }

  std::string get_ne_list_filename()          const { return ne_list_filename; } 
  std::string get_personnames_list_filename() const { return personnames_list_filename; } 
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
    else if (feat == "AllWNgrams")            f = AllWNgrams;
    else if (feat == "AllPOSBigrams")         f = AllT2grams;
    else if (feat == "AllPOSTrigrams")        f = AllT3grams;
    else if (feat == "AllPOSgrams")           f = AllTNgrams;
    else if (feat == "AllTokenTypes")         f = AllTokenTypes;
    else if (feat == "AllPersonNames")        f = AllPersonNames;
    else if (feat == "AllNamedEntities")      f = AllNamedEntities;
    else if (feat == "AllListFeatures")       f = AllListFeatures;
    else if (feat == "AllNELists")            f = AllNELists;
    else if (feat == "AllContextClues")       f = AllContextClues;
    else if (feat == "AllListFeatures")       f = AllListFeatures;
    else if (feat == "AllRegexes")            f = AllRegexes;
    else if (feat == "AllCharNgrams")         f = AllCharNgrams;
    else if (feat == "LeftContextContains")   f = LeftContextContains;
    else if (feat == "RoghtContextContains")  f = RightContextContains;
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

  bool true_bool_value(const std::string& v) const
  {
    if (v == "yes" || v == "true" || v == "1") return true;
    if (v == "no" || v == "false" || v == "0") return false;
    std::cerr << "NERConfiguration: invalid value '" << v << "'" << std::endl;
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
  std::string         ne_list_filename;
  std::string         personnames_list_filename;
  std::string         right_context_filename;
  std::string         left_context_filename;
  std::string         regex_filename;
  ColumnsMap          columns;
  bool                output_tok;
  bool                running_text_input;
  unsigned            order;
  unsigned            context_window_size;
}; // NERConfiguration

#endif
