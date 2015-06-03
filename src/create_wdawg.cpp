
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <algorithm>

#include <boost/tokenizer.hpp>

#include "../include/WDAWG.hpp"

typedef WeightedDirectedAcyclicWordGraph<std::string,std::string,
                                         StringUnsignedShortSerialiser>   StringWDAWG;

typedef StringWDAWG::EntryVector                                          EntryVector;
typedef StringWDAWG::Entry                                                Entry;
typedef std::vector<std::string>                                          TokenSeq;

void load_list(std::ifstream&, EntryVector&);
bool tokenize(const std::string&, TokenSeq&, unsigned);


int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: create_wdawg NE-LIST BIN_TRIE_FILE" << std::endl;
    exit(1);
  }

  std::ifstream list_in(argv[1]);
  if (!list_in) {
    std::cerr << "Error opening " << argv[1] << "\n";
    exit(2);
  }
  
  EntryVector list_entries;
  time_t t0 = clock();
  load_list(list_in,list_entries);
  time_t t1 = clock();
  
  StringWDAWG string_dawg(list_entries);
  time_t t2 = clock();
  std::cerr << "Constructed WDAWG: " 
            << string_dawg.no_of_states() << " states, " 
            << string_dawg.no_of_transitions() << " transitions, "
            << string_dawg.no_of_final_states() << " final states" << std::endl;

  //string_dawg.draw(std::ofstream("dawg.dot"));
  
  std::ofstream dawg_out(argv[2],std::ios::binary);
  if (!dawg_out) {
    std::cerr << "Error\n";
    exit(2);
  }
  
  string_dawg.write(dawg_out);
  dawg_out.close();
  time_t t3 = clock();
  
  std::cerr << "Wrote WDAWG to '" << argv[2] << "'" << std::endl;
  std::cerr << "Reading input list:  " << (t1-t0) << "ms" << std::endl;
  std::cerr << "Building DAWG:       " << (t2-t1) << "ms" << std::endl;
  std::cerr << "Writing binary file: " << (t3-t2) << "ms" << std::endl;

//  std::ifstream dawg_in(argv[2],std::ios::binary);
//  StringWDAWG string_dawg2(dawg_in);
//  string_dawg2.draw(std::ofstream("dawg2.dot"));
}

void load_list(std::ifstream& list_in, EntryVector& entries)
{
  std::string line;
  TokenSeq tokens;
  while (list_in.good()) {
    std::getline(list_in,line);
    if (!tokenize(line, tokens, 2))
      continue;
    entries.push_back(std::make_pair(TokenSeq(tokens.begin()+1,tokens.end()),tokens[0]));
  }
  std::sort(entries.begin(),entries.end());
}

bool tokenize(const std::string& line, TokenSeq& tokens, unsigned n)
{
  typedef boost::char_separator<char>     CharSeparator;
  typedef boost::tokenizer<CharSeparator> Tokenizer;
  
  Tokenizer tokenizer(line,CharSeparator("\t "));
  tokens.assign(tokenizer.begin(),tokenizer.end());
  // Check for Comment, empty line etc.
  return !(tokens.size() < n || (!tokens.empty() && tokens[0] == "#"));
}
