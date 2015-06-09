#ifndef __NER_OUTPUTTERS_HPP__
#define __NER_OUTPUTTERS_HPP__

#include <iostream>
#include <iterator>
#include <algorithm>

#include <boost/lexical_cast.hpp>

#include "TokenWithTag.hpp"

/// Base class for outputter function objects
struct NEROutputterBase
{
  virtual void prolog() {}
  virtual void epilog() {}
  /// Application mode
  virtual void operator()(const TokenWithTagSequence& sentence, bool last=false) {}
  virtual void operator()(const TokenWithTagSequence& sentence, const LabelSequence& inferred_labels, bool last=false) {}
}; // NEROutputterBase


/// Output for one word plus annotation per line on a stream
struct NEROneWordPerLineOutputter : public NEROutputterBase
{
  NEROneWordPerLineOutputter(std::ostream& o) : out(o) {}

  void prolog() {}
  void epilog() {}

  /// Application mode
  void operator()(const TokenWithTagSequence& sentence, bool last=false)
  {
    std::copy(sentence.begin(),sentence.end(),std::ostream_iterator<TokenWithTag>(out,"\n"));
    out << "\n";
  }

  /// Evaluation mode
  void operator()(const TokenWithTagSequence& sentence, const LabelSequence& inferred_labels, bool last=false)
  {
    for (unsigned i = 0; i < sentence.size(); ++i) {
      out << sentence[i].token << "\t" << sentence[i].position << "\t" << inferred_labels[i] << "\t" << sentence[i].label;
      out << ((inferred_labels[i] != sentence[i].label) ? "\t!!!\n" : "\n"); 
    }
    out << "\n";
  }

  std::ostream& out;
}; // NEROneWordPerLineOutputter


/// Output results as structured JSON output on a string stream
struct JSONOutputter : public NEROutputterBase
{
  JSONOutputter(std::ostream& o, bool pp=true) 
  : out(o), pretty_print(pp), entity_outputted(false) {}

  void prolog()
  {
    out << "{";
    if (pretty_print) out << std::endl;  
    out << (pretty_print ? "  " : "") << "\"entities\":[";
    if (pretty_print) out << std::endl;
  }

  void epilog()
  {
    if (pretty_print)
      out << std::endl << "  ]" << std::endl << "}" << std::endl;
    else 
      out << "]}";
  }

  /// Application mode
  void operator()(const TokenWithTagSequence& sentence, bool last=false)
  {
    std::string mwe, ne_type, ne_type_suff;
    unsigned ne_start_offset, ne_end_offset;
    bool in_ne = false;

    for (auto t = sentence.begin(); t != sentence.end(); ++t) {
      if (t->label == "OTHER") {
        if (in_ne) {
          // In the BIO annotation scheme, there's no L-marker
          output_ne(mwe,ne_type,ne_start_offset,ne_end_offset);
          mwe.clear();
          in_ne = false;
        }
      }
      else {
        // NE label
        ne_type = t->label.substr(0,t->label.size()-2);
        ne_type_suff = t->label.substr(t->label.size()-2);
        if (ne_type_suff == "_U") {
          output_ne(t->token,ne_type,t->position.offset,t->position.offset+t->position.length);
        }
        else {
          if (!in_ne) {
            if (ne_type_suff == "_B") {
              mwe = t->token;
              ne_start_offset = t->position.offset;
              in_ne = true;
            }
          }
          else {
            // With in ne
            if (ne_type_suff == "_L") {
              mwe = mwe + " " + t->token;
              output_ne(mwe,ne_type,ne_start_offset,t->position.offset+t->position.length);
              mwe.clear();
              in_ne = false;
            }
            else if (ne_type_suff == "_I") {
              mwe = mwe + " " + t->token;
              ne_end_offset = t->position.offset+t->position.length;
            }
          }
        }
      } 
    } // for t

    if (in_ne) {
      // Handle BIO annotation
      output_ne(mwe,ne_type,ne_start_offset,ne_end_offset);
    }
  }

  /// Evaluation mode
  void operator()(const TokenWithTagSequence& sentence, const LabelSequence& inferred_labels)
  {
  // Not yet
  }

private:
  void output_ne(const std::string& surface, const std::string& label, unsigned start, unsigned end) 
  {
    std::string indent((pretty_print ? 6 : 0), ' ');
    const char* double_quote = "\"";
    
    if (entity_outputted) out << ",";
    if (pretty_print) out << std::endl;
    out << std::string((pretty_print ? 4 : 0),' ') << "{";
    if (pretty_print) out << std::endl;
    output_key_val("surface",surface);
    output_key_val("entity_type",label);
    output_key_val("start",boost::lexical_cast<std::string>(start));
    output_key_val("end",boost::lexical_cast<std::string>(end),true);
    out << std::string((pretty_print ? 4 : 0),' ') << "}";
    entity_outputted = true;
  }

  void output_key_val(const std::string& key, const std::string& val, bool last=false)
  {
    const std::string indent((pretty_print ? 6 : 0), ' ');
    const char* double_quote = "\"";
    
    out << indent << double_quote << key << double_quote << ":";
    if (pretty_print) out << " ";
    out << double_quote << val << double_quote;
    if (!last) out << ",";
    if (pretty_print) out << std::endl;
  } 

private:
  std::ostream& out;      ///< Output stream
  bool pretty_print;      ///< Add indentation and newlines to the output
  bool entity_outputted;  ///< Used for placing syntactically correct commas in the JSON output
}; // JSONLineOutputter



/// Add XML-style annotation to running text
/// Note: this is a bit too much tailored towards NE recognition 
struct NERAnnotationOutputter : public NEROutputterBase
{
  NERAnnotationOutputter(std::ostream& o) : out(o) {}

  void prolog() {}
  void epilog() {}

  void operator()(const TokenWithTagSequence& sentence, bool last=false) const
  {
    bool in_ne = false;
    for (unsigned i = 0; i < sentence.size(); ++i) {
      const TokenWithTag& t = sentence[i];       
      if (t.label == "OTHER") {

        if (in_ne) {
          out << "</ne>";
          in_ne = false;
        }

        // Insert white space
        if (i > 0) {
          const TokenWithTag& prev_t = sentence[i-1];
          if (t.token_class == "PUNCT" || t.token_class == "R_QUOTE" || 
              t.token_class == "R_BRACKET" || t.token_class == "GENITIVE_SUFFIX") {
          }
          else {
            if (!(prev_t.token_class == "L_QUOTE" || prev_t.token_class == "L_BRACKET")) {
              out << " ";
            }
          }  
        } // if (i > 0)
        
        out << t.token;
      }
      else {
        // NE label
        std::string ne_type = t.label.substr(0,t.label.size()-2);
        std::string ne_type_suff = t.label.substr(t.label.size()-2);
        
        if (ne_type_suff == "_B") {
          if (i > 0) out << " ";
          out << "<ne class=\"" << ne_type << "\">" << t.token;
          in_ne = true;
        }
        else if (ne_type_suff == "_I") {
          out << " " << t.token;
        }
        else if (ne_type_suff == "_L") {
          out << " " << t.token;
        }
        else if (ne_type_suff == "_U") {
          if (i > 0) out << " ";
          out << "<ne class=\"" << ne_type << "\">" << t.token << "</ne>";
        }
      }
    }
    out << "\n";
  }

  void operator()(const TokenWithTagSequence& sentence, const LabelSequence& inferred_labels, bool last=false) const 
  {
  }

  std::ostream& out;
}; // NERAnnotationOutputter

#endif
