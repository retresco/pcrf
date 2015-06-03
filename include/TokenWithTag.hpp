////////////////////////////////////////////////////////////////////////////////////////////////////
// TokenWithTag.hpp
// Token representation
// TH, Oct. 2014
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __TOKENWITHTAG_HPP__
#define __TOKENWITHTAG_HPP__

#include <string>
#include <vector>
#include <iostream>

#include "tokenizer.hpp"
  
/// Represents a text token together with its tokenizer class, POS tag, label and position
struct TokenWithTag
{
  TokenWithTag(const std::string& tok) : token(tok) {}
  TokenWithTag(const std::string& tok, const std::string& tc) 
  : token(tok), token_class(tc) {}
  TokenWithTag(const std::string& tok, const Tokenizer::TokenPosition& pos) 
  : token(tok), position(pos) {}
  TokenWithTag(const std::string& tok, const std::string& tc, const Tokenizer::TokenPosition& pos) 
  : token(tok), token_class(tc), position(pos) {}
    
  void assign_label(const std::string& l) { label = l; }
  void assign_tag(const std::string& t) { tag = t; }
  void assign_chunk(const std::string& ch) { chunk = ch; }

  friend std::ostream& operator<<(std::ostream& o, const TokenWithTag& wt)
  {
    if (!wt.label.empty()) o << wt.label << "\t";
    o << wt.token << "\t";
    if (!wt.token_class.empty()) o << wt.token_class << "\t";
    if (wt.position.valid()) o << wt.position;
    return o;
  }  
  
  std::string token;                    ///< The text token
  std::string lemma;
  std::string token_class;              ///< Its tokenizer class
  std::string tag;                      ///< Optional: tag
  std::string label;                    ///< Optional: label (assigned by training data)
  std::string chunk;                    ///< Optional: chunk
  Tokenizer::TokenPosition position;    ///< Position in text
}; // TokenWithTag

/// Represents a input sequence of annotated tokens
typedef std::vector<TokenWithTag>  TokenWithTagSequence;

#endif
