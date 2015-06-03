////////////////////////////////////////////////////////////////////////////////
// NERFeatureExtractor.hpp
// TH, Sept. 2014
// Feature extractor for CRF named entity recognition system
////////////////////////////////////////////////////////////////////////////////

#ifndef __NER_FEATURE_EXTRACTOR_HPP__
#define __NER_FEATURE_EXTRACTOR_HPP__

/* TODO
  - Error in Token classification in case of punctuations etc.
  - Merge CharNGrams and prefixes/suffixes?
  - Possible feature: PrevToken-NextToken
  - shape sequences
  - Mask :
*/

//#define USE_BOOST_REGEX

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <bitset>
#include <cctype>

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#ifdef USE_BOOST_REGEX
  #include <boost/regex.hpp>
#endif

#include "CRFTypedefs.hpp"
#include "WDAWG.hpp"
#include "AsyncTokenizer.hpp"
#include "TokenWithTag.hpp"

//#include <boost/multiprecision/cpp_int.hpp>


/// 64-bit number with bit positions encoding features
typedef unsigned long long FeatureType;
//typedef boost::multiprecision::int128_t FeatureType;

/// Available features: numbers encode bit positions within FeatureType
const unsigned FWord                    = 0;      ///< Current word
const unsigned FWord_p1                 = 1;      ///< Previous word
const unsigned FWord_p2                 = 2;      ///< Previous2 word
const unsigned FWord_n1                 = 3;      ///< Next word
const unsigned FWord_n2                 = 4;      ///< Next2 word

const unsigned FWordLowerCased          = 5;      ///< Current word in lowercase

const unsigned FPosT                    = 6;      ///< Current tag
const unsigned FPosT_p1                 = 7;      ///< Previous tag
const unsigned FPosT_p2                 = 8;      ///< Previous2 tag
const unsigned FPosT_n1                 = 9;      ///< Next tag
const unsigned FPosT_n2                 = 10;     ///< Next2 tag

const unsigned FLemma                   = 11;     ///< Current lemma
const unsigned FLemma_p1                = 12;     ///< Previous lemma
const unsigned FLemma_p2                = 13;     ///< Previous2 lemma
const unsigned FLemma_n1                = 14;     ///< Next lemma
const unsigned FLemma_n2                = 15;     ///< Next2 lemma

const unsigned FTokenClass              = 16;     ///< Tokenizer class

// Delimiter features
const unsigned FBos                     = 17;     /// Begin of sequence
const unsigned FEos                     = 18;     /// End of sequence

// Word-Ngram features
const unsigned FW2grams_l               = 19;     ///< Word bigram left
const unsigned FW2grams_r               = 20;     ///< Word bigram right

const unsigned FW3grams_l               = 21;     ///< Word trigram left
const unsigned FW3grams_c               = 22;     ///< Word trigram center
const unsigned FW3grams_r               = 23;     ///< Word trigram right

// Tag-Ngram features
const unsigned FPOS2grams_l             = 24;     ///< Tag bigram left
const unsigned FPOS2grams_r             = 25;     ///< Tag bigram right

const unsigned FPOS3grams_l             = 26;     ///< Tag trigram left
const unsigned FPOS3grams_c             = 27;     ///< Tag trigram center
const unsigned FPOS3grams_r             = 28;     ///< Tag trigram right

// Token-tag features
const unsigned FWordPOS                 = 29;     ///< Word-Tag

// Prefix features
const unsigned FPrefW1                  = 30;     ///< Word prefix of length 1
const unsigned FPrefW2                  = 31;     ///< Word prefix of length 2
const unsigned FPrefW3                  = 32;     ///< Word prefix of length 3
const unsigned FPrefW4                  = 33;     ///< Word prefix of length 4
// Suffix features
const unsigned FSuffW1                  = 34;     ///< Word suffix of length 1
const unsigned FSuffW2                  = 35;     ///< Word suffix of length 2
const unsigned FSuffW3                  = 36;     ///< Word suffix of length 3
const unsigned FSuffW4                  = 37;     ///< Word suffix of length 4

// Word classification features
const unsigned FAllUpper                = 38;
const unsigned FAllDigit                = 39;
const unsigned FAllSymbol               = 40;
const unsigned FAllUpperOrDigit         = 41;
const unsigned FAllUpperOrSymbol        = 42;
const unsigned FAllDigitOrSymbol        = 43;
const unsigned FAllUpperOrDigitOrSymbol = 44;
const unsigned FInitUpper               = 45;
const unsigned FAllLetter               = 46;
const unsigned FAllAlnum                = 47;

const unsigned FInitUpper2g_l           = 48;
const unsigned FInitUpper2g_r           = 49;
const unsigned FInitUpper3g_l           = 50;
const unsigned FInitUpper3g_c           = 41;
const unsigned FInitUpper3g_r           = 52;

const unsigned FTokenShape              = 53;
const unsigned FVCPattern               = 54;
const unsigned FCharNgrams              = 55;


// Features based on lists
const unsigned FListPersonName          = 56;
const unsigned FListNamedEntity         = 57;
const unsigned FLeftContextClues        = 58;
const unsigned FRightContextClues       = 59;
const unsigned FRegex                   = 60;

const unsigned FLeftContextContains     = 61;
const unsigned FRightContextContains    = 62;

// 
#define SETFEAT(f)                      (FeatureType(1) << FeatureType(f))
#define FEAT_VAL_SEP                    std::string("=")
#define NGRAM_SEP                       std::string("|")

// Common feature combinations
const FeatureType HeadWord              = SETFEAT(FWord);
const FeatureType HeadWordLowercased    = SETFEAT(FWordLowerCased);


const FeatureType AllPrevWords          = SETFEAT(FWord_p1)|SETFEAT(FWord_p2);
const FeatureType AllNextWords          = SETFEAT(FWord_n1)|SETFEAT(FWord_n2);
const FeatureType AllWords              = HeadWord|AllPrevWords|AllNextWords;

const FeatureType AllPosTags            = SETFEAT(FPosT)|SETFEAT(FPosT_p1)|SETFEAT(FPosT_p2)|
                                          SETFEAT(FPosT_n1)|SETFEAT(FPosT_n2);

const FeatureType WordPOS               = SETFEAT(FWordPOS);
const FeatureType TokenClass            = SETFEAT(FTokenClass);
const FeatureType AllLemmas             = SETFEAT(FLemma)|SETFEAT(FLemma_p1)|SETFEAT(FLemma_p2)|
                                          SETFEAT(FLemma_n1)|SETFEAT(FLemma_n2);
const FeatureType AllPrefixes           = SETFEAT(FPrefW1)|SETFEAT(FPrefW2)|SETFEAT(FPrefW3)|SETFEAT(FPrefW4);
const FeatureType AllSuffixes           = SETFEAT(FSuffW1)|SETFEAT(FSuffW2)|SETFEAT(FSuffW3)|SETFEAT(FSuffW4);
const FeatureType AllDelim              = SETFEAT(FBos)|SETFEAT(FEos);

const FeatureType AllW2grams            = SETFEAT(FW2grams_l)|SETFEAT(FW2grams_r);
const FeatureType AllW3grams            = SETFEAT(FW3grams_l)|SETFEAT(FW3grams_c)|SETFEAT(FW3grams_r);
const FeatureType AllWNgrams            = AllW2grams|AllW3grams;

const FeatureType AllT2grams            = SETFEAT(FPOS2grams_l)|SETFEAT(FPOS2grams_r);
const FeatureType AllT3grams            = SETFEAT(FPOS3grams_l)|SETFEAT(FPOS3grams_c)|SETFEAT(FPOS3grams_r);
const FeatureType AllTNgrams            = AllT2grams|AllT3grams;

const FeatureType AllTokenTypes         = SETFEAT(FAllUpper)|SETFEAT(FAllDigit)|SETFEAT(FAllSymbol)|
                                          SETFEAT(FAllUpperOrDigit)|SETFEAT(FAllUpperOrSymbol)|
                                          SETFEAT(FAllDigitOrSymbol)|SETFEAT(FAllUpperOrDigitOrSymbol)|
                                          SETFEAT(FInitUpper)|SETFEAT(FAllLetter)|SETFEAT(FAllAlnum);

const FeatureType AllInitUpper2grams    = SETFEAT(FInitUpper2g_l)|SETFEAT(FInitUpper2g_r);
const FeatureType AllInitUpper3grams    = SETFEAT(FInitUpper3g_l)|SETFEAT(FInitUpper3g_c)|SETFEAT(FInitUpper3g_r);
const FeatureType AllInitUpperGrams     = AllInitUpper2grams|AllInitUpper3grams;


const FeatureType AllContextClues       = SETFEAT(FLeftContextClues)|SETFEAT(FRightContextClues);
const FeatureType AllRegexes            = SETFEAT(FRegex);
const FeatureType AllCharNgrams         = SETFEAT(FCharNgrams);

const FeatureType LeftContextContains   = SETFEAT(FLeftContextContains);
const FeatureType RightContextContains  = SETFEAT(FRightContextContains);
const FeatureType AllContextContains    = LeftContextContains|RightContextContains;

const FeatureType AllNELists            = SETFEAT(FListNamedEntity);
const FeatureType AllPersonNames        = SETFEAT(FListPersonName);

const FeatureType AllNamedEntities      = AllNELists; //AllPersonNames|
const FeatureType AllListFeatures       = AllContextClues|AllNELists|AllPersonNames|AllPersonNames;
const FeatureType AllShapes             = SETFEAT(FTokenShape);
const FeatureType VCPattern             = SETFEAT(FVCPattern);

const FeatureType AllFeatures           = AllPrefixes|AllSuffixes|AllWords|AllPosTags|WordPOS|AllLemmas|AllDelim|
                                          AllW2grams|AllW3grams|AllT2grams|AllT3grams|AllTokenTypes|AllContextClues|
                                          AllListFeatures|
                                          AllRegexes|AllCharNgrams|AllContextContains|AllShapes|VCPattern|TokenClass;


/// Names of the features (used for outputting attributes)
static const char* FeatureNames[] = {
  "W[0]",                               // 0
  "W[-1]",                              // 1
  "W[-2]",                              // 2
  "W[1]",                               // 3
  "W[2]",                               // 4
  "lcW[0]",                             // 5
  "POS[0]",                             // 6
  "POS[-1]",                            // 7
  "POS[-2]",                            // 8
  "POS[1]",                             // 9
  "POS[2]",                             // 10
  "L[0]",                               // 11
  "L[-1]",                              // 12
  "L[-2]",                              // 13
  "L[1]",                               // 14
  "L[2]",                               // 15
  "TokClass",                           // 16
  "<BOS>",                              // 17
  "<EOS>",                              // 18
  "W[-1]|W[0]",                         // 19
  "W[0]|W[1]",                          // 20
  "W[-2]|W[-1]|W[0]",                   // 21
  "W[-1]|W[0]|W[1]",                    // 22
  "W[0]|W[1]|W[2]",                     // 23
  "POS[-1]|POS[0]",                     // 24
  "POS[0]|POS[1]",                      // 25
  "POS[-2]|POS[-1]|POS[0]",             // 26
  "POS[-1]|POS[0]|POS[1]",              // 27
  "POS[0]|POS[1]|POS[2]",               // 28
  "Word|POS",                           // 29
  "Pref1",                              // 30
  "Pref2",                              // 31
  "Pref3",                              // 32
  "Pref4",                              // 33
  "Suff1",                              // 34
  "Suff2",                              // 35
  "Suff3",                              // 36
  "Suff4",                              // 37
  "AllUpper",                           // 38
  "AllDigit",                           // 39
  "AllSymbol",                          // 40
  "AllUpperOrDigit",                    // 41
  "AllUpperOrSymbol",                   // 42
  "AllDigitOrSymbol",                   // 43
  "AllUpperOrDigitOrSymbol",            // 44
  "InitUpper",                          // 45
  "AllLetter",                          // 46
  "AllAlnum",                           // 47
  "InitCap[-1]|InitCap[0]",             // 48
  "InitCap[0]|InitCap[1]",              // 49
  "InitCap[-2]|InitCap[-1]|InitCap[0]", // 50
  "InitCap[-1]|InitCap[0]|InitCap[1]",  // 51
  "InitCap[0]|InitCap[1]|InitCap[2]",   // 52
  "Shape",                              // 53
  "VC",                                 // 54
  "CharNgram",                          // 55
  "PossiblePersonName",                 // 56
  "ListNE",                             // 57
  "LC-NE-Clue",                         // 58
  "RC-NE-Clue",                         // 59
  "Regex",                              // 60
  "InLC",                               // 61
  "InRC"                                // 62
 }; // FeatureNames


/// Annotation scheme for labeling
typedef enum { nerBIO, nerBILOU }   NERAnnotationScheme;


/// NERFeatureExtractor
class NERFeatureExtractor
{
public: // Types
  typedef std::bitset<64>                                     GeneratedFeatures;
  typedef AsyncTokenizer::TokenPosition                       TokenPosition;
  typedef enum { ngrams_left, ngrams_center, ngrams_right }   NGramDir;
  
public:
  NERFeatureExtractor(FeatureType gf=AllFeatures, bool have_tags=false, 
                      unsigned n1=3, unsigned n2=4, unsigned n3=8) 
  : data_contains_tags(have_tags), max_ngram_width(n1), 
    max_char_ngram_width(n2), max_context_range(n3)
  {
    for (unsigned f = 0; f < (sizeof(FeatureNames)/sizeof(FeatureNames[0])); ++f) {
      if (gf & (FeatureType(1) << f)) gen_feat[f] = true;
    }
    //std::cerr << gf << "\n";
    //std::cerr << gen_feat << "\n";
    //std::cerr << max_context_range << "\n";
  }
  
  ~NERFeatureExtractor()
  {}

  /// Set the window size for context features
  void set_context_window_size(unsigned r)
  {
    max_context_range = r;
  }

  CRFInputSequence add_features(const TokenWithTagSequence& x) const
  {
    CRFInputSequence iseq;

    for (unsigned t = 0; t < x.size(); ++t) {
      const TokenWithTag& x_t = x[t];
      AttributeVector as;
      if (!x_t.label.empty()) as.push_back(x_t.label); // TODO: BUG!
      check_and_add_features(x,t,as);
      iseq.push_back(WordWithAttributes(x_t.token,as));
    }
    
    if (gen_feat.test(FListPersonName)) 
      add_list_features(x,FListPersonName,person_names_dawg,iseq);

    if (gen_feat.test(FListNamedEntity)) 
      add_list_features(x,FListNamedEntity,ne_dawg,iseq);

    if (gen_feat.test(FLeftContextClues)) 
      add_context_clues(x,FLeftContextClues,left_context_dawg,iseq);

    if (gen_feat.test(FRightContextClues)) 
      add_context_clues(x,FRightContextClues,right_context_dawg,iseq);

    return iseq;
  }
    
  void add_ne_list(std::ifstream& in)
  {
    ne_dawg.read(in);
  }

  void add_person_names_list(std::ifstream& in)
  {
    person_names_dawg.read(in);
    //person_names_dawg.draw(std::ofstream("person_names.dot"));
  }

  void add_left_context_list(std::ifstream& in)
  {
    add_context_list(in,left_context_dawg);
  }

  void add_right_context_list(std::ifstream& in)
  {
    add_context_list(in,right_context_dawg);
  }

  void add_word_regex_list(std::ifstream& in)
  {
    std::string line;
    TokenSeq tokens;
    while (in.good()) {
      std::getline(in,line);
      if (!tokenize(line, tokens, 2))
        continue;
      add_word_regex(tokens[1],tokens[0]);
    }
  }

  /// Inform the extractor about the presence/absence of POS tag information in the training data
  void have_pos_tags(bool v)
  {
    data_contains_tags = v;
  }

private:
  // Token classification constants
  static const unsigned AllUpper                  = 0;
  static const unsigned AllDigit                  = 1;
  static const unsigned AllSymbol                 = 2;
  static const unsigned AllUpperOrDigit           = 3;
  static const unsigned AllUpperOrSymbol          = 4;
  static const unsigned AllDigitOrSymbol          = 5;
  static const unsigned AllUpperOrDigitOrSymbol   = 6;
  static const unsigned InitUpper                 = 7;
  static const unsigned AllLetter                 = 8;
  static const unsigned AllAlnum                  = 9;
  
  typedef WeightedDirectedAcyclicWordGraph<std::string,std::string,
                                           StringUnsignedShortSerialiser>     StringDAWG;
  typedef std::bitset<10>                                                     TokenTypeFeat;
  typedef StringDAWG                                                          NamesDAWG;
  typedef StringDAWG                                                          ContextDAWG;
  typedef std::vector<std::string>                                            TokenSeq;
  typedef NamesDAWG::State                                                    DAWGState;
  typedef NamesDAWG::FinalStateInfoSet                                        DAWGStateInfoSet;

#ifdef USE_BOOST_REGEX
  typedef std::map<std::string,boost::regex>  Regexes;
#endif

private:
  void check_and_add_features(const TokenWithTagSequence& x, unsigned t, AttributeVector& as) const
  {
    if (gen_feat.test(FWord)) 
      add_feature(FeatureNames[FWord],mask(x[t].token),false,as);

    if (gen_feat.test(FWordLowerCased)) 
      add_feature(FeatureNames[FWordLowerCased],mask(lowercase(x[t].token)),false,as);

    if (gen_feat.test(FTokenShape)) 
      add_feature(FeatureNames[FTokenShape],shape(x[t].token),false,as);

    if (gen_feat.test(FTokenClass)) 
      add_feature(FeatureNames[FTokenClass],x[t].token_class,false,as);

    if (gen_feat.test(FVCPattern)) 
      add_feature(FeatureNames[FVCPattern],sound_pattern(x[t].token),false,as);

    if (gen_feat.test(FWord_p1) && t > 0) 
      add_feature(FeatureNames[FWord_p1],mask(x[t-1].token),false,as);

    if (gen_feat.test(FWord_p2) && t > 1) 
      add_feature(FeatureNames[FWord_p2],mask(x[t-2].token),false,as);

    if (gen_feat.test(FWord_n1) && int(t) < int(x.size())-1) 
      add_feature(FeatureNames[FWord_n1],mask(x[t+1].token),false,as);

    if (gen_feat.test(FWord_n2) && int(t) < int(x.size())-2) 
      add_feature(FeatureNames[FWord_n2],mask(x[t+2].token),false,as);

    if (data_contains_tags) {
      if (gen_feat.test(FPosT)) add_feature(FeatureNames[FPosT],x[t].tag,false,as);
      if (gen_feat.test(FPosT_p1) && t > 0) add_feature(FeatureNames[FPosT_p1],x[t-1].tag,false,as);
      if (gen_feat.test(FPosT_p2) && t > 1) add_feature(FeatureNames[FPosT_p2],x[t-2].tag,false,as);
      if (gen_feat.test(FPosT_n1) && int(t) < int(x.size())-1) add_feature(FeatureNames[FPosT_n1],x[t+1].tag,false,as);
      if (gen_feat.test(FPosT_n2) && int(t) < int(x.size())-2) add_feature(FeatureNames[FPosT_n2],x[t+2].tag,false,as);
      if (gen_feat.test(FPosT_p1) && t > 0) add_feature(FeatureNames[FPosT_p1],x[t-1].tag,false,as);
    }

    // N-grams
    if (gen_feat.test(FW2grams_l)) add_token_ngrams(x,t,2,ngrams_left,FW2grams_l,as);
    if (gen_feat.test(FW2grams_r)) add_token_ngrams(x,t,2,ngrams_right,FW2grams_r, as);

    if (gen_feat.test(FW3grams_l)) add_token_ngrams(x,t,3,ngrams_left,FW3grams_l,as);
    if (gen_feat.test(FW3grams_c)) add_token_ngrams(x,t,3,ngrams_center,FW3grams_c,as);
    if (gen_feat.test(FW3grams_r)) add_token_ngrams(x,t,3,ngrams_right,FW3grams_r,as);


    // InitUpper N-grams
    if (gen_feat.test(FInitUpper2g_l)) add_tokentypes_ngrams(x,t,FInitUpper2g_l,as);
    if (gen_feat.test(FInitUpper2g_r)) add_tokentypes_ngrams(x,t,FInitUpper2g_r, as);

    if (gen_feat.test(FInitUpper3g_l)) add_tokentypes_ngrams(x,t,FInitUpper3g_l,as);
    if (gen_feat.test(FInitUpper3g_c)) add_tokentypes_ngrams(x,t,FInitUpper3g_c,as);
    if (gen_feat.test(FInitUpper3g_r)) add_tokentypes_ngrams(x,t,FInitUpper3g_r,as);


    // Tag sequences
    if (data_contains_tags) {
      if (gen_feat.test(FPOS2grams_l)) 
        add_pos_ngrams(x,t,2,ngrams_left,FPOS2grams_l,as);
      if (gen_feat.test(FPOS2grams_r)) 
        add_pos_ngrams(x,t,2,ngrams_right,FPOS2grams_l,as);

      if (gen_feat.test(FPOS3grams_l)) 
        add_pos_ngrams(x,t,3,ngrams_left,FPOS3grams_l,as);
      if (gen_feat.test(FPOS3grams_c)) 
        add_pos_ngrams(x,t,3,ngrams_center,FPOS3grams_c,as);
      if (gen_feat.test(FPOS3grams_r)) 
        add_pos_ngrams(x,t,3,ngrams_right,FPOS3grams_r,as);
    }

    // Word-POS pairs
    if (gen_feat.test(FWordPOS) && data_contains_tags) 
      add_feature(FeatureNames[FWordPOS],mask(x[t].token)+NGRAM_SEP+x[t].tag,false,as);

    // Prefixes
    if (gen_feat.test(FPrefW1)) add_feature(FeatureNames[FPrefW1],mask(prefix(x[t].token,1)),false,as);
    if (gen_feat.test(FPrefW2)) add_feature(FeatureNames[FPrefW2],mask(prefix(x[t].token,2)),false,as);
    if (gen_feat.test(FPrefW3)) add_feature(FeatureNames[FPrefW3],mask(prefix(x[t].token,3)),false,as);
    if (gen_feat.test(FPrefW4)) add_feature(FeatureNames[FPrefW4],mask(prefix(x[t].token,4)),false,as);

    // Suffixes
    if (gen_feat.test(FSuffW1)) add_feature(FeatureNames[FSuffW1],mask(suffix(x[t].token,1)),false,as);
    if (gen_feat.test(FSuffW2)) add_feature(FeatureNames[FSuffW2],mask(suffix(x[t].token,2)),false,as);
    if (gen_feat.test(FSuffW3)) add_feature(FeatureNames[FSuffW3],mask(suffix(x[t].token,3)),false,as);
    if (gen_feat.test(FSuffW4)) add_feature(FeatureNames[FSuffW4],mask(suffix(x[t].token,4)),false,as);

    // Token type features
    TokenTypeFeat tt = get_type(x[t].token);
    if (gen_feat.test(FAllUpper) && tt.test(AllUpper)) 
      add_feature(FeatureNames[FAllUpper],"",true,as);
    if (gen_feat.test(FAllDigit) && tt.test(AllDigit)) 
      add_feature(FeatureNames[FAllDigit],"",true,as);
    if (gen_feat.test(FAllSymbol) && tt.test(AllSymbol)) 
      add_feature(FeatureNames[FAllSymbol],"",true,as);
    if (gen_feat.test(FAllUpperOrDigit) && tt.test(AllUpperOrDigit)) 
      add_feature(FeatureNames[FAllUpperOrDigit],"",true,as);
    if (gen_feat.test(FAllUpperOrSymbol) && tt.test(AllUpperOrSymbol)) 
      add_feature(FeatureNames[FAllUpperOrSymbol],"",true,as);
    if (gen_feat.test(FAllDigitOrSymbol) && tt.test(AllDigitOrSymbol)) 
      add_feature(FeatureNames[FAllDigitOrSymbol],"",true,as);
    if (gen_feat.test(FAllUpperOrDigitOrSymbol) && tt.test(AllUpperOrDigitOrSymbol)) 
      add_feature(FeatureNames[FAllUpperOrDigitOrSymbol],"",true,as);
    if (gen_feat.test(FInitUpper) && tt.test(InitUpper))  
      add_feature(FeatureNames[FInitUpper],"",true,as);
    if (gen_feat.test(FAllLetter) && tt.test(AllLetter)) 
      add_feature(FeatureNames[FAllLetter],"",true,as);
    if (gen_feat.test(FAllAlnum) && tt.test(AllAlnum)) 
      add_feature(FeatureNames[FAllAlnum],"",true,as);

    // Regex tests
    if (gen_feat.test(FRegex)) add_regex_features(x[t],as);
    
    if (gen_feat.test(FCharNgrams) && x[t].token.size() > 1) 
      add_char_ngram_features(x[t].token,as);
    
    if (gen_feat.test(FLeftContextContains))
      add_left_context_words(x,t,as);
    if (gen_feat.test(FRightContextContains))
      add_right_context_words(x,t,as);

    if (gen_feat.test(FBos) && t == 0) add_feature(FeatureNames[FBos],"",true,as);
    if (gen_feat.test(FEos) && t == x.size()-1) add_feature(FeatureNames[FEos],"",true,as);
  }

  void add_feature(const std::string& feat, const std::string& val, bool unary, AttributeVector& as) const
  {
    //std::cerr << feat << "=" << val << std::endl;
    if (!val.empty()) as.push_back(feat_val(feat,val));
    else if (unary) as.push_back(feat);
  }

  void add_word_regex(const std::string& re, const std::string& name)
  {
#ifdef USE_BOOST_REGEX
    regexes.insert(std::make_pair(name,boost::regex(re)));
#endif
  }

  void add_regex_features(const TokenWithTag& x, AttributeVector& as) const
  {
#ifdef USE_BOOST_REGEX
    for (Regexes::const_iterator r = regexes.begin(); r != regexes.end(); ++r) {
      if (boost::regex_match(x.token,r->second)) {
        add_feature(FeatureNames[FRegex],r->first,false,as);
      }
    }
#endif
  }
  
  void add_context_list(std::ifstream& in, ContextDAWG& dawg)
  {
    dawg.read(in);
  }

  void add_token_ngrams(const TokenWithTagSequence& x, unsigned t, unsigned ngram_width, NGramDir dir, 
                        unsigned feat_index, AttributeVector& as) const
  {
    const std::string ng_feat = FeatureNames[feat_index];

    if (ngram_width == 2) {
      if (dir == ngrams_left) {
        if (t > 0) {
          std::string ng_val = mask(x[t-1].token) + NGRAM_SEP + mask(x[t].token);
          add_feature(ng_feat,ng_val,false,as);
        }
      }
      else if (dir == ngrams_right) {
        if (t < x.size()-1) {
          std::string ng_val = mask(x[t].token) + NGRAM_SEP + mask(x[t+1].token);
          add_feature(ng_feat,ng_val,false,as);
        }
      }
    }
    else if (ngram_width == 3) {
      if (dir == ngrams_left) {
        if (t > 1) {
          std::string ng_val = mask(x[t-2].token) + NGRAM_SEP + mask(x[t-1].token) + NGRAM_SEP + mask(x[t].token);
          add_feature(ng_feat,ng_val,false,as);
        }
      }
      else if (dir == ngrams_center) {
        if (t > 0 && t < x.size()-1) {
          std::string ng_val = mask(x[t-1].token) + NGRAM_SEP + mask(x[t].token) + NGRAM_SEP + mask(x[t+1].token);
          add_feature(ng_feat,ng_val,false,as);
        }
      }
      else if (dir == ngrams_right) {
        if (int(t) < int(x.size())-2) {
          std::string ng_val = mask(x[t].token) + NGRAM_SEP + mask(x[t+1].token) + NGRAM_SEP + mask(x[t+2].token);
          add_feature(ng_feat,ng_val,false,as);
        }
      }
    }
  }

  void add_tokentypes_ngrams(const TokenWithTagSequence& x, unsigned t, unsigned feat_index, AttributeVector& as) const
  {
    const std::string ng_feat = FeatureNames[feat_index];

    if (feat_index == FInitUpper2g_l) {
      if (t > 0 && init_upper(x[t-1].token) && init_upper(x[t].token)) {
        add_feature(ng_feat,"",true,as);
      }
    }
    else if (feat_index == FInitUpper2g_r) {
      if (t < x.size()-1 && init_upper(x[t].token) && init_upper(x[t+1].token)) {
        add_feature(ng_feat,"",true,as);
      }
    }
    else if (feat_index == FInitUpper3g_l) {
      if (t > 1 && init_upper(x[t-2].token) && init_upper(x[t-1].token) && init_upper(x[t].token)) {
        add_feature(ng_feat,"",true,as);
      }
    }
    else if (feat_index == FInitUpper3g_c) {
      if (t > 0 && t < x.size()-1 && init_upper(x[t-1].token) && init_upper(x[t].token) && init_upper(x[t+1].token)) {
        add_feature(ng_feat,"",true,as);
      }
    }
    else if (feat_index == FInitUpper3g_r) {
      if (int(t) < int(x.size())-2 && init_upper(x[t].token) && init_upper(x[t+1].token) && init_upper(x[t+2].token)) {
        add_feature(ng_feat,"",true,as);
      }
    }
  }

  void add_pos_ngrams(const TokenWithTagSequence& x, unsigned t, unsigned ngram_width, NGramDir dir, 
                      unsigned feat_index, AttributeVector& as) const
  {
    const std::string ng_feat = FeatureNames[feat_index];
    if (ngram_width == 2) {
      if (dir == ngrams_left) {
        if (t > 0) {
          std::string ng_val = x[t-1].tag + NGRAM_SEP + x[t].tag;
          add_feature(ng_feat,ng_val,false,as);
        }
      }
      else if (dir == ngrams_right) {
        if (t < x.size()-1) {
          std::string ng_val = x[t].tag + NGRAM_SEP + x[t+1].tag;
          add_feature(ng_feat,ng_val,false,as);
        }
      }
    }
    else if (ngram_width == 3) {
      if (dir == ngrams_left) {
        if (t > 1) {
          std::string ng_val = x[t-2].tag + NGRAM_SEP + x[t-1].tag + NGRAM_SEP + x[t].tag;
          add_feature(ng_feat,ng_val,false,as);
        }
      }
      else if (dir == ngrams_center) {
        if (t > 0 && t < x.size()-1) {
          std::string ng_val = x[t-1].tag + NGRAM_SEP + x[t].tag + NGRAM_SEP + x[t+1].tag;
          add_feature(ng_feat,ng_val,false,as);
        }
      }
      else if (dir == ngrams_right) {
        if (int(t) < int(x.size())-2) {
          std::string ng_val = x[t].tag + NGRAM_SEP + x[t+1].tag + NGRAM_SEP + x[t+2].tag;
          add_feature(ng_feat,ng_val,false,as);
        }
      }
    }
  }

  template<typename DAWG>
  void add_list_features(const TokenWithTagSequence& x, unsigned f, DAWG& dawg, CRFInputSequence& iseq) const
  {
    typedef typename DAWG::State               DAWGState;
    typedef typename DAWG::FinalStateInfoSet   DAWGStateInfoSet;

    for (unsigned t = 0; t < x.size(); ++t) {
      const TokenWithTag& x_t = x[t];
      DAWGState q = dawg.start_state();
      for (unsigned t1 = t; t1 < x.size(); ++t1) {
        // Check whether current word starts a Wiki name
        DAWGState p = dawg.find_transition(q,x[t1].token);
        if (p == DAWG::NoState()) break; // No transition found
        if (dawg.is_final(p)) {
          // Wiki name found => get annotation
          const DAWGStateInfoSet& dawg_entries = dawg.final_info(p);
          // Iterate over the annotations
          for (auto e = dawg_entries.begin(); e != dawg_entries.end(); ++e) {
            // Iterate over the span covered by the NE and add features
            for (int k = t; k <= t1; ++k) {
              std::string feat = std::string(FeatureNames[f]) + "[" +
                                 boost::lexical_cast<std::string>(int(t)-k) +
                                 ".." + boost::lexical_cast<std::string>(int(t1)-k) + "]";
              iseq[k].attributes.push_back(feat_val(feat,*e));
            } // for k
          } // for e
        } // if
        // Continue finding even longer names
        q = p;
      } // for
    } // for t
  }

  template<typename DAWG>
  void add_context_clues(const TokenWithTagSequence& x, unsigned f, DAWG& dawg, CRFInputSequence& iseq) const
  {
    typedef typename DAWG::State               DAWGState;
    typedef typename DAWG::FinalStateInfoSet   DAWGStateInfoSet;

    bool to_the_right = (f == FLeftContextClues);
    for (unsigned t = 0; t < x.size(); ++t) {
      const TokenWithTag& x_t = x[t];
      DAWGState q = dawg.start_state();
      for (unsigned t1 = t; t1 < x.size(); ++t1) {
        // Check whether current word starts a Wiki name
        DAWGState p = dawg.find_transition(q,x[t1].token);
        if (p == DAWG::NoState()) break; // No transition found
        if (dawg.is_final(p)) {
          // Wiki name found => get annotation
          const DAWGStateInfoSet& dawg_entries = dawg.final_info(p);
          if (to_the_right && (t1 < iseq.size()-1)) {
            // Target word is to the right
            for (auto e = dawg_entries.begin(); e != dawg_entries.end(); ++e) {
              add_feature(FeatureNames[f],*e,false,iseq[t1+1].attributes);
            }
          }
          else if (!to_the_right && t > 0) {
            // Target word is to the left
            for (auto e = dawg_entries.begin(); e != dawg_entries.end(); ++e) {
              add_feature(FeatureNames[f],*e,false,iseq[t-1].attributes);
            }
          }
        } // if
        // Continue finding even longer names
        q = p;
      } // for
    } // for t
  }

  void add_char_ngram_features(const std::string& xt, AttributeVector& as) const
  {
    for (unsigned n = 2; n <= std::min(max_char_ngram_width,unsigned(xt.size())); ++n) {
      for (unsigned i = 0; i <= xt.size()-n; ++i) {
        std::string ng_feat = std::string(FeatureNames[FCharNgrams]) + "[" +
                              boost::lexical_cast<std::string>(i) +
                              ".." + boost::lexical_cast<std::string>(i+n-1) + "]";
        add_feature(ng_feat,mask(xt.substr(i,n)),false,as);
      }
    }
  }

  /// Adds features like InLC[-4..0]=company
  void add_left_context_words(const TokenWithTagSequence& x, unsigned t, AttributeVector& as) const
  {
    for (int n = 1; n <= max_context_range; ++n) {
      if (int(t)-n < 0) break;
      std::string feat = std::string(FeatureNames[FLeftContextContains]) + "[" +
                         boost::lexical_cast<std::string>(-int(max_context_range)) +
                         "..0]";
      add_feature(feat,mask(x[int(t)-n].token),false,as);
    }
  }

  /// Adds features like InRC[0..4]=company
  void add_right_context_words(const TokenWithTagSequence& x, unsigned t, AttributeVector& as) const
  {
    for (int n = 1; n <= max_context_range; ++n) {
      if (t+n >= x.size()) break;
      std::string feat = std::string(FeatureNames[FRightContextContains]) + "[0" +
                         ".." + boost::lexical_cast<std::string>(max_context_range) + "]";
      add_feature(feat,mask(x[t+n].token),false,as);
    }
  }

  std::string feat_val(const std::string& feat, const std::string& val) const
  {
    return feat + FEAT_VAL_SEP + val;
  }
  
  std::string prefix(const std::string& w, unsigned n) const
  {
    return (w.size() >= n) ? w.substr(0,n) : "";
  }

  std::string suffix(const std::string& w, unsigned n) const
  {
    return (w.size() >= n) ? w.substr(w.size()-n) : "";
  }

  TokenTypeFeat get_type(const std::string& token) const
  {
    TokenTypeFeat r;
    if (token.empty()) return r;
    r.set();
    if (!std::isupper(token[0])) r[InitUpper] = false; 
    
    for (unsigned i = 0; i < token.size(); ++i) {
      char c = token[i];
      if (std::isupper(c)) {
        r[AllDigit] = r[AllSymbol] = r[AllDigitOrSymbol] = false; 
      }
      else if (std::isdigit(c) || c == ',' || c == '.') {
        r[AllUpper] = r[AllSymbol] = r[AllUpperOrSymbol] = r[AllLetter] = false; 
      }
      else if (std::islower(c)) {
        r[AllUpper] = r[AllDigit] = r[AllSymbol] = r[AllUpperOrDigit] = false; 
        r[AllUpperOrSymbol] = r[AllDigitOrSymbol] = r[AllUpperOrDigitOrSymbol] = false; 
      }
      else {
        r[AllUpper] = r[AllDigit] = r[AllUpperOrDigit] = r[AllLetter] = r[AllAlnum] = false; 
      }
    }
    return r;
  }
  
  ///
  std::string lowercase(const std::string& tok) const
  {
    std::string tok2(tok);
    for (auto i = 0; i < tok2.size(); tok2[i] = std::tolower(tok2[i]), ++i);
    //std::transform(tok2.begin(), tok2.end(), tok2.begin(), std::tolower);
    return tok2;
  }

  /// Extracts a token shape: X for uppercase letters, 9 for digits etc.
  std::string shape(const std::string& tok) const
  {
    std::string s;
    for (std::string::const_iterator c = tok.begin(); c != tok.end(); ++c) {
      if (std::isalpha(*c) && std::isupper(*c)) s += 'X';
      else if (std::isalpha(*c) && std::islower(*c)) s += 'x';
      else if (std::isdigit(*c)) s += '9';
      else if (*c == '-') s += '-';
      else if (*c == '.') s += '.';
      else s += '#';
    }
    return s;
  }

  std::string sound_pattern(const std::string& tok) const
  {
    std::string s;
    for (std::string::const_iterator c = tok.begin(); c != tok.end(); ++c) {
      if (std::isalpha(*c)) s += is_vowel(*c) ? 'V' : 'C';
      else if (std::isdigit(*c)) s += '9';
      else if (*c == '-') s += '-';
      else if (*c == '.') s += '.';
      else s += '#';
    }
    return s;
  }

  /// Currently replaces only : (for crfsuite training)
  std::string mask(const std::string& tok) const
  {
    if (tok.find_first_of(":") == std::string::npos) return tok;  
    std::string s;
    for (std::string::const_iterator c = tok.begin(); c != tok.end(); ++c) {
      if (*c == ':') s += "__COLON__";
      else s += *c;
    }
    return s;
  }

  bool is_vowel(char c) const
  {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
           c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
  }

  bool init_upper(const std::string& s) const
  {
    if (s.empty()) return false;
    return std::isupper(s[0]);
  }


  bool tokenize(const std::string& line, TokenSeq& tokens, unsigned n) const
  {
    typedef boost::char_separator<char>     CharSeparator;
    typedef boost::tokenizer<CharSeparator> Tokenizer;
    
    Tokenizer tokenizer(line,CharSeparator("\t "));
    tokens.assign(tokenizer.begin(),tokenizer.end());
    // Check for Comment, empty line etc.
    return !(tokens.size() < n || (!tokens.empty() && tokens[0] == "#"));
  }
  
private:
  GeneratedFeatures   gen_feat;               ///< Feature flags determining which features are generated
  unsigned            order;                  ///< Markov order
  unsigned            max_ngram_width;        ///< Max. width of word/tag/... N-grams
  unsigned            max_char_ngram_width;   ///< Max. width of word char N-grams
  unsigned            max_context_range;      ///< Window size of "occurs in left/right context" feature
  bool                data_contains_tags;     ///< Does the training data contain tags
  NamesDAWG           ne_dawg;                ///< DAWG for Wiki names etc.
  NamesDAWG           person_names_dawg;      ///< DAWG for first and last names
  ContextDAWG         left_context_dawg;      ///< DAWG for left context clues
  ContextDAWG         right_context_dawg;     ///< DAWG for right context clues
#ifdef USE_BOOST_REGEX
  Regexes             regexes;                ///< Regexes to match against the input token
#endif
}; // NERFeatureAttributer

#endif
