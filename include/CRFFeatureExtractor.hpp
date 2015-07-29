////////////////////////////////////////////////////////////////////////////////
// CRFFeatureExtractor.hpp
// TH, July 2014
// Feature extractor for CRF related annotation tasks
// Supersedes NERFeatureExtractor
////////////////////////////////////////////////////////////////////////////////

#ifndef __CRF_FEATURE_EXTRACTOR_HPP__
#define __CRF_FEATURE_EXTRACTOR_HPP__

/* TODO
  - Error in Token classification in case of punctuations etc.
  - Merge CharNGrams and prefixes/suffixes?
  - Possible feature: PrevToken-NextToken
  - Inner N-grams
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
const unsigned FW2grams                 = 19;     ///< Word bigram
const unsigned FW3grams                 = 20;     ///< Word trigram
const unsigned FW4grams                 = 21;     ///< Word quadgram
const unsigned FW5grams                 = 22;     ///< Word pentagram
const unsigned FW6grams                 = 23;     ///< Word 6-gram
const unsigned FW7grams                 = 24;     ///< Word 7-gram
const unsigned FW8grams                 = 25;     ///< Word 8-gram
const unsigned FW9grams                 = 26;     ///< Word 9-gram
const unsigned FW10grams                = 27;     ///< Word 10-gram

// Tag-Ngram features
const unsigned FPOS2grams               = 28;     ///< Tag bigram
const unsigned FPOS3grams               = 29;     ///< Tag trigram
const unsigned FPOS4grams               = 30;     ///< Tag quadgram

// Token-tag features
const unsigned FWordPOS                 = 31;     ///< Word-Tag

// Prefix features
const unsigned FPrefW                   = 32;     ///< Word prefix
// Suffix features
const unsigned FSuffW                   = 33;     ///< Word suffix

// Word classification features
const unsigned FAllUpper                = 34;
const unsigned FAllDigit                = 35;
const unsigned FAllSymbol               = 36;
const unsigned FAllUpperOrDigit         = 37;
const unsigned FAllUpperOrSymbol        = 38;
const unsigned FAllDigitOrSymbol        = 39;
const unsigned FAllUpperOrDigitOrSymbol = 40;
const unsigned FInitUpper               = 41;
const unsigned FAllLetter               = 42;
const unsigned FAllAlnum                = 43;

const unsigned FInitUpper2g             = 44;
const unsigned FInitUpper3g             = 45;

const unsigned FTokenShape              = 46;
const unsigned FVCPattern               = 47;
const unsigned FCharNgrams              = 48;


// Features based on lists
//const unsigned FListPersonName          = 49;
const unsigned FPatternsList            = 50;
const unsigned FLeftContextClues        = 51;
const unsigned FRightContextClues       = 52;
const unsigned FRegex                   = 53;

const unsigned FLeftContextContains     = 54;
const unsigned FRightContextContains    = 55;

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
const FeatureType AllPrefixes           = SETFEAT(FPrefW);
const FeatureType AllSuffixes           = SETFEAT(FSuffW);
const FeatureType AllDelim              = SETFEAT(FBos)|SETFEAT(FEos);

const FeatureType AllW2grams            = SETFEAT(FW2grams);
const FeatureType AllW3grams            = SETFEAT(FW3grams);
const FeatureType AllW4grams            = SETFEAT(FW4grams);
const FeatureType AllW5grams            = SETFEAT(FW5grams);
const FeatureType AllW6grams            = SETFEAT(FW6grams);
const FeatureType AllW7grams            = SETFEAT(FW7grams);
const FeatureType AllW8grams            = SETFEAT(FW8grams);
const FeatureType AllW9grams            = SETFEAT(FW9grams);
const FeatureType AllW10grams           = SETFEAT(FW10grams);
const FeatureType AllWNgrams            = AllW2grams|AllW3grams;

const FeatureType AllT2grams            = SETFEAT(FPOS2grams);
const FeatureType AllT3grams            = SETFEAT(FPOS3grams);
const FeatureType AllTNgrams            = AllT2grams|AllT3grams;

const FeatureType AllTokenTypes         = SETFEAT(FAllUpper)|SETFEAT(FAllDigit)|SETFEAT(FAllSymbol)|
                                          SETFEAT(FAllUpperOrDigit)|SETFEAT(FAllUpperOrSymbol)|
                                          SETFEAT(FAllDigitOrSymbol)|SETFEAT(FAllUpperOrDigitOrSymbol)|
                                          SETFEAT(FInitUpper)|SETFEAT(FAllLetter)|SETFEAT(FAllAlnum);

const FeatureType AllInitUpper2grams    = SETFEAT(FInitUpper2g);
const FeatureType AllInitUpper3grams    = SETFEAT(FInitUpper3g);
const FeatureType AllInitUpperGrams     = AllInitUpper2grams|AllInitUpper3grams;


const FeatureType AllContextClues       = SETFEAT(FLeftContextClues)|SETFEAT(FRightContextClues);
const FeatureType AllRegexes            = SETFEAT(FRegex);
const FeatureType AllCharNgrams         = SETFEAT(FCharNgrams);

const FeatureType LeftContextContains   = SETFEAT(FLeftContextContains);
const FeatureType RightContextContains  = SETFEAT(FRightContextContains);
const FeatureType AllContextContains    = LeftContextContains|RightContextContains;

const FeatureType AllPatterns           = SETFEAT(FPatternsList);
//const FeatureType AllPersonNames        = SETFEAT(FListPersonName);

//const FeatureType AllNamedEntities      = AllNELists; //AllPersonNames|
const FeatureType AllListFeatures       = AllContextClues|AllPatterns;
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
  "W",                                  // 19
  "W",                                  // 20
  "W",                                  // 21
  "W",                                  // 22
  "W",                                  // 23
  "W",                                  // 24
  "W",                                  // 25
  "W",                                  // 26
  "W",                                  // 27
  "POS",                                // 28
  "POS",                                // 29
  "POS",                                // 30
  "W|POS",                              // 31
  "Pref",                               // 32
  "Suff",                               // 33
  "AllUpper",                           // 34
  "AllDigit",                           // 35
  "AllSymbol",                          // 36
  "AllUpperOrDigit",                    // 37
  "AllUpperOrSymbol",                   // 38
  "AllDigitOrSymbol",                   // 39
  "AllUpperOrDigitOrSymbol",            // 40
  "InitUpper",                          // 41
  "AllLetter",                          // 42
  "AllAlnum",                           // 43
  "InitCap",                            // 44
  "InitCap",                            // 45
  "Shape",                              // 46
  "VC",                                 // 47
  "CharNgram",                          // 48
  "PossiblePersonName",                 // 49
  "PatternClass",                       // 50
  "LC-Clue",                            // 51
  "RC-Clue",                            // 52
  "Regex",                              // 53
  "InLC",                               // 54
  "InRC"                                // 55
 }; // FeatureNames


/// Annotation scheme for labeling
typedef enum { nerBIO, nerBILOU }   NERAnnotationScheme;


/// CRFFeatureExtractor implements CRF feature annotation
class CRFFeatureExtractor
{
public: // Types
  typedef std::bitset<64>                                     GeneratedFeatures;
  typedef AsyncTokenizer::TokenPosition                       TokenPosition;
  typedef enum { ngrams_left, ngrams_center, ngrams_right }   NGramDir;
  
public:
  /**
    @brief Constructor
    @param Features to extract (encoded as a bit vector)
  */
  CRFFeatureExtractor(FeatureType gf=AllFeatures, bool have_tags=false, 
                      unsigned n1=3, unsigned n2=4, unsigned n3=8) 
  : data_contains_tags(have_tags), max_ngram_width(n1), 
    max_char_ngram_width(n2), max_context_range(n3), add_inner_ngrams(false),
    max_word_prefix_len(4), max_word_suffix_len(4) 
  {
    for (unsigned f = 0; f < (sizeof(FeatureNames)/sizeof(FeatureNames[0])); ++f) {
      if (gf & (FeatureType(1) << f)) gen_feat[f] = true;
    }
    //std::cerr << gf << "\n";
    //std::cerr << gen_feat << "\n";
    //std::cerr << max_context_range << "\n";
  }
  
  /// Set the window size for context features
  void set_context_window_size(unsigned r)
  {
    max_context_range = r;
  }
  
  /// In true, inner word N-grams are generated as features
  void set_inner_word_ngrams(bool v)
  {
    add_inner_ngrams = v;
  }

  /**
    @brief Adds features to a sequence x
    @param seq Input sequence which holds the tokens and a number of additional properties
    @return a sequence which consists out of pairs (t,av) where t is the input token
            and av is a vector of strings where each string represents an annotated
            feature
  */
  CRFInputSequence add_features(const TokenWithTagSequence& seq) const
  {
    CRFInputSequence iseq;
    iseq.reserve(seq.size());
    AttributeVector as;

    for (unsigned t = 0; t < seq.size(); ++t) {
      const TokenWithTag& seq_t = seq[t];
      as.clear();
      if (!seq_t.label.empty()) as.push_back(seq_t.label); // TODO: BUG!
      // Add feature strings to as
      check_and_add_features(seq,t,as);
      iseq.push_back(WordWithAttributes(seq_t.token,as));
    }
    
//    if (gen_feat.test(FListPersonName)) 
//      add_list_features(x,FListPersonName,person_names_dawg,iseq);

    if (gen_feat.test(FPatternsList)) 
      add_list_features(seq,FPatternsList,patterns_dawg,iseq);

    if (gen_feat.test(FLeftContextClues)) 
      add_context_clues(seq,FLeftContextClues,left_context_dawg,iseq);

    if (gen_feat.test(FRightContextClues)) 
      add_context_clues(seq,FRightContextClues,right_context_dawg,iseq);

    return iseq;
  }
  
  /// Add the DAWG entries in the binary stream 'in' to the feature extractor 
  void add_patterns(std::ifstream& in)
  {
    patterns_dawg.read(in);
  }

//  void add_person_names_list(std::ifstream& in)
//  {
//    person_names_dawg.read(in);
//    //person_names_dawg.draw(std::ofstream("person_names.dot"));
//  }

  /// Add left context DAWG
  void add_left_contexts(std::ifstream& in)
  {
    add_contexts(in,left_context_dawg);
  }

  /// Add right context DAWG
  void add_right_contexts(std::ifstream& in)
  {
    add_contexts(in,right_context_dawg);
  }

  /// Add regexes from a two column text file
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
  /// Token classification constants
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
  
  /// Instantiation of the word graph
  typedef WeightedDirectedAcyclicWordGraph<std::string,std::string,
                                           StringUnsignedShortSerialiser>     StringDAWG;
  typedef std::bitset<10>                                                     TokenTypeFeat;
  typedef StringDAWG                                                          PatternsDAWG;
  typedef StringDAWG                                                          ContextDAWG;
  typedef std::vector<std::string>                                            TokenSeq;
  typedef PatternsDAWG::State                                                 DAWGState;
  typedef PatternsDAWG::FinalStateInfoSet                                     DAWGStateInfoSet;

#ifdef USE_BOOST_REGEX
  typedef std::map<std::string,boost::regex>                                  Regexes;
#endif

private:
  /// Work horse: adds all features related to position t in x to as
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
    if (gen_feat.test(FW2grams)) {
      add_token_ngrams(x,t,2,ngrams_left,FW2grams,as);
      add_token_ngrams(x,t,2,ngrams_right,FW2grams,as);
    }

    for (unsigned k = 1; k < 9; ++k) {
      if (gen_feat.test(FW2grams+k)) {
        add_token_ngrams(x,t,k+2,ngrams_left,FW2grams+k,as);
        if (add_inner_ngrams) {
          add_token_ngrams(x,t,k+2,ngrams_center,FW2grams+k,as);
        }
        add_token_ngrams(x,t,k+2,ngrams_right,FW2grams+k,as);
      }
    } // for k

    // InitUpper N-grams
    //if (gen_feat.test(FInitUpper2g_l)) add_tokentypes_ngrams(x,t,FInitUpper2g_l,as);
    //if (gen_feat.test(FInitUpper2g_r)) add_tokentypes_ngrams(x,t,FInitUpper2g_r, as);

    //if (gen_feat.test(FInitUpper3g_l)) add_tokentypes_ngrams(x,t,FInitUpper3g_l,as);
    //if (gen_feat.test(FInitUpper3g_c)) add_tokentypes_ngrams(x,t,FInitUpper3g_c,as);
    //if (gen_feat.test(FInitUpper3g_r)) add_tokentypes_ngrams(x,t,FInitUpper3g_r,as);


    // Tag sequences
    if (data_contains_tags) {
      if (gen_feat.test(FPOS2grams)) { 
        add_pos_ngrams(x,t,2,ngrams_left,FPOS2grams,as);
        add_pos_ngrams(x,t,2,ngrams_right,FPOS2grams,as);
      }

      if (gen_feat.test(FPOS3grams)) { 
        add_pos_ngrams(x,t,3,ngrams_left,FPOS3grams,as);
        add_pos_ngrams(x,t,3,ngrams_center,FPOS3grams,as);
        add_pos_ngrams(x,t,3,ngrams_right,FPOS3grams,as);
      }
    }

    // Word-POS pairs
    if (gen_feat.test(FWordPOS) && data_contains_tags) 
      add_feature(FeatureNames[FWordPOS],mask(x[t].token)+NGRAM_SEP+x[t].tag,false,as);

    // Prefixes
    if (gen_feat.test(FPrefW)) {
      for (unsigned l = 1; l <= max_word_prefix_len; ++l) {
        add_feature(FeatureNames[FPrefW],mask(prefix(x[t].token,l)),false,as);
      }
    }

    // Suffixes
    if (gen_feat.test(FSuffW)) {
      for (unsigned l = 1; l <= max_word_suffix_len; ++l) {
        add_feature(FeatureNames[FSuffW],mask(suffix(x[t].token,l)),false,as);
      }
    }

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
  
  void add_contexts(std::ifstream& in, ContextDAWG& dawg)
  {
    dawg.read(in);
  }

  void add_token_ngrams(const TokenWithTagSequence& x, unsigned t, 
                        unsigned ngram_width, NGramDir dir, 
                        unsigned feat_index, AttributeVector& as) const
  {
    std::string token_pref("W");
    if (dir == ngrams_left && t >= ngram_width-1) {
      add_feature(make_ngram_feat(token_pref,t,t-ngram_width+1,ngram_width),
                  make_ngram(t-ngram_width+1,t,x),false,as);
    }
    else if (dir == ngrams_right && t+ngram_width-1 < x.size()) {
      add_feature(make_ngram_feat(token_pref,t,t,ngram_width),
                  make_ngram(t,t+ngram_width-1,x),false,as);
    }
    else if (dir == ngrams_center && 
             ngram_width > 2 && int(t)-ngram_width+2 >= 0 && t+ngram_width-2 < x.size()) {
      for (unsigned start = t-ngram_width+2; start < t; ++start) {
        add_feature(make_ngram_feat(token_pref,t,start,ngram_width),
                    make_ngram(start,start+ngram_width-1,x),false,as);
      } // for k
    }
  }

  void add_tokentypes_ngrams(const TokenWithTagSequence& x, unsigned t, unsigned feat_index, AttributeVector& as) const
  {
    const std::string ng_feat = FeatureNames[feat_index];

    //if (feat_index == FInitUpper2g_l) {
    //  if (t > 0 && init_upper(x[t-1].token) && init_upper(x[t].token)) {
    //    add_feature(ng_feat,"",true,as);
    //  }
    //}
    //else if (feat_index == FInitUpper2g_r) {
    //  if (t < x.size()-1 && init_upper(x[t].token) && init_upper(x[t+1].token)) {
    //    add_feature(ng_feat,"",true,as);
    //  }
    //}
    //else if (feat_index == FInitUpper3g_l) {
    //  if (t > 1 && init_upper(x[t-2].token) && init_upper(x[t-1].token) && init_upper(x[t].token)) {
    //    add_feature(ng_feat,"",true,as);
    //  }
    //}
    //else if (feat_index == FInitUpper3g_c) {
    //  if (t > 0 && t < x.size()-1 && init_upper(x[t-1].token) && init_upper(x[t].token) && init_upper(x[t+1].token)) {
    //    add_feature(ng_feat,"",true,as);
    //  }
    //}
    //else if (feat_index == FInitUpper3g_r) {
    //  if (int(t) < int(x.size())-2 && init_upper(x[t].token) && init_upper(x[t+1].token) && init_upper(x[t+2].token)) {
    //    add_feature(ng_feat,"",true,as);
    //  }
    //}
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
  
private: // Functions
  //template<typename PROJECTOR>
  std::string make_ngram(unsigned from, unsigned to, const TokenWithTagSequence& x) const
  {
    std::string result = mask(x[from].token);
    for (unsigned k = from+1; k <= to; ++k) {
      result = result + NGRAM_SEP + mask(x[k].token);
    }
    return result;
  }  

  /// Build the feature name of an N-gram feature  
  std::string make_ngram_feat(const std::string& pref, unsigned t, unsigned start, unsigned width) const
  {
    int from = start-t;
    return pref + "[" + boost::lexical_cast<std::string>(from) + ".." +
           boost::lexical_cast<std::string>(from+width-1) + "]";
  }  

private:
  GeneratedFeatures   gen_feat;               ///< Feature flags determining which features are generated
  unsigned            order;                  ///< Markov order
  unsigned            max_ngram_width;        ///< Max. width of word/tag/... N-grams
  bool                add_inner_ngrams;       ///< If true, all overlapping N-grams at a given position are added
  unsigned            max_char_ngram_width;   ///< Max. width of word char N-grams
  unsigned            max_context_range;      ///< Window size of "occurs in left/right context" feature
  unsigned            max_word_prefix_len;
  unsigned            max_word_suffix_len;
  bool                data_contains_tags;     ///< Does the training data contain tags
  PatternsDAWG        patterns_dawg;          ///< DAWG for Wiki names etc.
  PatternsDAWG        person_names_dawg;      ///< DAWG for first and last names
  ContextDAWG         left_context_dawg;      ///< DAWG for left context clues
  ContextDAWG         right_context_dawg;     ///< DAWG for right context clues
#ifdef USE_BOOST_REGEX
  Regexes             regexes;                ///< Regexes to match against the input token
#endif
}; // CRFFeatureExtractor

#endif
