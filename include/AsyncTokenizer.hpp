#ifndef __ASYNCTOKENIZER_HPP__
#define __ASYNCTOKENIZER_HPP__

#include <string>
#include <iostream>

#include "tokenizer.hpp"
#include "TokenWithTag.hpp"

/// Implements a tokeniser which extracts a input sequences (sentences) from a running text file
class AsyncTokenizer
{
public:
  /// Contains position information about a token
  typedef Tokenizer::TokenPosition    TokenPosition;

public:
  /**
    @brief Constructor
    @param in the text stream (in ISO or UTF-8)
    @param eas if true the BLIOU scheme is used for annotation (instead of the BIO scheme)
    @param o model order (1 or 2) 
    @param def_class Default annotation label 
    @note The model order param is deprecated.
  */
  AsyncTokenizer(std::istream& in, bool eas, unsigned o, const std::string& dl) 
  : text_in(in), enhanced_annotation_scheme(eas), order(o), tok_count(0), ne_seq_begin(false), 
    current_line_processed(true), default_label(dl), current_ne_class(dl)
  {}

  /// Asychroniously tokenize the input text, return sentence by sentence
  /// Precondition: sentence must be emptied before each call to tokenise()
  bool tokenize(TokenWithTagSequence& sentence);

  /// Returns the total token count so far
  unsigned total_token_count() const { return tok_count; }

private:
  /// Asychroniously tokenize the input text, return sentence by sentence
  /// Precondition: sentence must be emptied before each call to tokenise()
  bool tokenize2(TokenWithTagSequence& sentence);

  void change_annotation(TokenWithTagSequence& sentence) const;

  // Hack: generalize that!
  std::string extract_ne_class(const std::string& t) const
  {
    if (t == "<ne class=\"PER\">" || t == "<ne class=\\\"PER\\\">") return "PER";
    if (t == "<ne class=\"ORG\">" || t == "<ne class=\\\"ORG\\\">") return "ORG";
    if (t == "<ne class=\"PRO\">" || t == "<ne class=\\\"PRO\\\">") return "PRO";
    if (t == "<ne class=\"EVE\">" || t == "<ne class=\\\"EVE\\\">") return "EVE";
    if (t == "<ne class=\"LOC\">" || t == "<ne class=\\\"LOC\\\">") return "LOC";
    return "UNK";
  }

  std::string build_label(const std::string& current_ne_class, const std::string& prev_ne_class) const
  {
    switch (order) {
      case 1:   return current_ne_class;
      case 2:   return prev_ne_class + "-" + current_ne_class;
      default:  return current_ne_class;
    }
  }

private:
  typedef Tokenizer::Token  Token;

private:
  std::istream& text_in;
  bool          enhanced_annotation_scheme;   ///< Use BILOU instead of BIO
  Tokenizer     tokenizer;                    ///<
  unsigned      tok_count;
  bool          ne_seq_begin;   
  unsigned      order;                        ///<
  std::string   current_line;
  std::string   current_ne_class;
  std::string   default_label;
  bool          current_line_processed;
}; // AsyncTokenizer


bool AsyncTokenizer::tokenize(TokenWithTagSequence& sentence)
{
  std::string prev_enhanced_ne_class = "BOS";
  std::string enhanced_ne_class;

  if (current_line_processed) {
    // Get a new text line
    if (!text_in.good()) {
      return false;
    }
    std::getline(text_in,current_line);
    tokenizer.set_line(current_line.c_str());
    current_line_processed = false;
  }

  for (Token t = tokenizer.next_token(); t != Tokenizer::ttEOS; t = tokenizer.next_token()) {
    std::string tok = t.token();
    ++tok_count;
    if (t == Tokenizer::ttNEAnnotation) {
      // Start of an annotation found
      current_ne_class = extract_ne_class(t.token());
      ne_seq_begin = true; 
    }
    else if (t == Tokenizer::ttNEAnnotationEnd) {
      // End of an annotation found
      current_ne_class = default_label;
      ne_seq_begin = false;
    }
    else {
      // Normal token
      if (current_ne_class == default_label) {
        enhanced_ne_class = default_label;
      }
      else {
        // If we already encountered a <ne class=??>, determine whether it's the first element
        // in the NE sequence or a subsequent element (they get different classes)
        if (ne_seq_begin) {
          ne_seq_begin = false;
          if (enhanced_annotation_scheme) {
            // BILOU annotation scheme
            const Token& t_lookahead = tokenizer.lookahead();
            if (t_lookahead == Tokenizer::ttNEAnnotationEnd)
              // unit NE found
              enhanced_ne_class = current_ne_class + "_U";
            else
              enhanced_ne_class = current_ne_class + "_B";
          }
          else {
            // BIO annotation scheme
            enhanced_ne_class = current_ne_class + "_B";
          }
        }
        else {
          // Inside a NE sequence
          if (enhanced_annotation_scheme) {
            // BILOU annotation scheme
            const Token& t_lookahead = tokenizer.lookahead();
            if (t_lookahead == Tokenizer::ttNEAnnotationEnd)
              // unit NE found
              enhanced_ne_class = current_ne_class + "_L";
            else
              enhanced_ne_class = current_ne_class + "_I";
          }
          else {
            // BIO annotation scheme
            enhanced_ne_class = current_ne_class + "_I";
          }
        }  
      }

      TokenWithTag to(t.token(),tokenizer.translation(t.type()),t.position());
      to.assign_label(build_label(enhanced_ne_class,prev_enhanced_ne_class));
      sentence.push_back(to);
      prev_enhanced_ne_class = enhanced_ne_class;

      // If current token ends a sentence, return complete sequence
      if (t == Tokenizer::ttPunct && 
          (t.token() == "." || t.token() == "!" || t.token() == "?" )) {
        const Token& t_lookahead = tokenizer.lookahead();
        if (t_lookahead == Tokenizer::ttRightQuote && 
            t_lookahead.position().offset == t.position().offset+1) {
          // Adjacent closing quote found => consume it, add it to the sequence and return
          // .” or .’, presumably quoted sequence
          t = tokenizer.next_token();
          TokenWithTag to(t.token(),tokenizer.translation(t.type()),t.position());
          to.assign_label(default_label);
          sentence.push_back(to);
          prev_enhanced_ne_class = default_label; 
        }
        return true;
      } // if (t == Tokenizer::ttPunct
    }
  } // for t

  current_line_processed = true;
  return true;
}

/// A bit of a hack
void AsyncTokenizer::change_annotation(TokenWithTagSequence& sentence) const
{
  for (unsigned i = 0; i < sentence.size(); ++i) {
    std::string& label = sentence[i].label;
    if (label != default_label) {
      std::string suffix = label.substr(label.size()-2);
      if (suffix == "_B" && 
        ((i == sentence.size()-1) || sentence[i+1].label == default_label || sentence[i+1].label.substr(sentence[i+1].label.size()-2) == "_B")) {
        // Next label is 'OTHER' or current is the last in the sequence
        label = label.substr(0,label.size()-2) + "_U";
      }
      if (suffix == "_I" && ((i == sentence.size()-1) || sentence[i+1].label == default_label)) {
        // Next label is 'OTHER' or current is the last in the sequence
        label = label.substr(0,label.size()-2) + "_L";
      }
    }
  }
}

#endif
