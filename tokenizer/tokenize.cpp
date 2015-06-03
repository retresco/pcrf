////////////////////////////////////////////////////////////////////////////////
// tokenizer.cpp
// Tokenizer demo
// TH, 8.7.2013
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include "tokenizer.hpp"

std::string extract_ne_class(const std::string& t);
  
int main()
{
  typedef Tokenizer::Token  Token;
  
  std::string line;
  Tokenizer tokenizer;
  std::string current_ne_class = "OTHER";
  bool ne_seq_begin = false;
  
  while (std::cin.good()) {
    std::getline(std::cin,line);
    tokenizer.set_line(line.c_str());
    for (Token t = tokenizer.next_token(); t != Tokenizer::ttEOS; t = tokenizer.next_token()) {
      if (t == Tokenizer::ttNEAnnotation) {
        // Start of an annotation found
        current_ne_class = extract_ne_class(t.token());
        ne_seq_begin = true; 
      }
      else if (t == Tokenizer::ttNEAnnotationEnd) {
        // End of an annotation found
        current_ne_class = "OTHER";
        ne_seq_begin = false;
      }
      else {
        // Normal token
        std::string enhanced_ne_class;
        if (current_ne_class == "OTHER") enhanced_ne_class = "OTHER";
        else {
          // If we already encountered a <ne class=??>, determine whether it's the first element
          // in the NE sequence or a subsequent element (they get different classes)
          if (ne_seq_begin) {
            enhanced_ne_class = current_ne_class + "_B";
            ne_seq_begin = false;
          }
          else {
            enhanced_ne_class = current_ne_class + "_I";
          }  
        }
        // Output NE class, token, token class and position
        std::cout << enhanced_ne_class << "\t" << t.token() << "\t" 
                  << tokenizer.translation(t.type()) << "\t" 
                  << t.position() << "\n";
        if (t == Tokenizer::ttPunct && 
            (t.token() == "." || t.token() == "!" || t.token() == "?" )) 
          std::cout << std::endl;
      }
    }
  }
}

// Hack: generalize that!
std::string extract_ne_class(const std::string& t)
{
  if (t == "<ne class=\"PER\">") return "PER";
  if (t == "<ne class=\"ORG\">") return "ORG";
  if (t == "<ne class=\"PRO\">") return "PRO";
  if (t == "<ne class=\"EVE\">") return "EVE";
  if (t == "<ne class=\"LOC\">") return "LOC";
  return "UNK";
}
