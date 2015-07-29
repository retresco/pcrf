#ifndef __CRFTRAININGCORPUS_HPP__
#define __CRFTRAININGCORPUS_HPP__

#include <vector>
#include <algorithm>
#include <iostream>

#include <boost/tokenizer.hpp>
#include <boost/unordered_map.hpp>

#include "CRFTypedefs.hpp"
#include "StringUnsignedMapper.hpp"
#include "CRFDecoder.hpp"
#include "SimpleLinearCRFModel.hpp"


/** 
  @brief CRFTranslatedTrainingCorpus represents a translated corpus, that is, a sequence
  of n pairs (x,y), where x is the input sequence consisting out of the input tokens
  and their translated attributes, and y is translated label sequence.
*/
class CRFTranslatedTrainingCorpus
{
private:
  typedef boost::unordered_map<unsigned,unsigned>                 FeatureCountMap;

public:
  /// Creates an instance of a translated corpus and reserves room for n training pairs
  CRFTranslatedTrainingCorpus(unsigned n=0) 
  : max_len(0), tok_count(0), attr_counter(0), label_counter(0), token_type_counter(0)
  {
    training_pairs.reserve(n);
    training_pairs_indices.reserve(n);
    map_label("<BOS>");
  }

  /// Constructor from istream associated with a tab-separated text file
  CRFTranslatedTrainingCorpus(std::istream& corpus_in)
  : max_len(0), tok_count(0), attr_counter(0), label_counter(0), token_type_counter(0)
  {
    map_label("<BOS>");
    read(corpus_in);
  }

  /// Clears the training corpus and returns all memory
  void clear()
  {
    std::vector<TranslatedCRFTrainingPair>().swap(training_pairs);
    std::vector<unsigned>().swap(training_pairs_indices);
    attributes_mapper.clear();
    labels_mapper.clear();
    feature_counts.clear();
    max_len = tok_count = attr_counter = label_counter = 0;
  }

  /// Returns the corpus size
  unsigned size() const { return training_pairs.size(); }

  /// Returns the size of the longest input sequence of a training pair
  unsigned max_input_length() const { return max_len; }

  /// Return the ID of the dummy label BOS
  LabelID get_bos_label() const { return 0; }

  /// Returns a const reference to the training pair at position index
  inline const TranslatedCRFTrainingPair& operator[](unsigned index) const
  {
    static TranslatedCRFTrainingPair invalid;
    return (index < size()) ? training_pairs[training_pairs_indices[index]] : invalid;
  }

  /// Returns a non-const reference to the training pair at position index
  inline TranslatedCRFTrainingPair& operator[](unsigned index)
  {
    static TranslatedCRFTrainingPair invalid;
    return (index < size()) ? training_pairs[training_pairs_indices[index]] : invalid;
  }

  /// Append an untranslated training pair tp to the corpus; tp will be translated
  void add(const CRFTrainingPair& tp)
  {
    if (tp.first.size() == tp.second.size()) {
      TranslatedCRFTrainingPair ttp;
      ttp.x.resize(tp.first.size());
      ttp.y.resize(tp.second.size());
      AttributeIDVector attributes;
      for (unsigned i = 0; i < tp.second.size(); ++i) {
        // Map attributes
        attributes.clear();
        for (unsigned a = 0; a < tp.first[i].attributes.size(); ++a) {
          attributes.push_back(map_attr(tp.first[i].attributes[a]));
        }
        AttributeIDVector(attributes).swap(attributes);
        ttp.x[i] = boost::make_tuple(map_token(tp.first[i].token),attributes);
        ttp.y[i] = map_label(tp.second[i]);
      } // for i
      add(ttp);
    }
    else {
      std::cerr << "Error: input and output vectors are of different lengths." << std::endl;
    }
  }

  /// Append an translated training pair tp to the corpus  
  void add(const TranslatedCRFTrainingPair& tp) 
  {
    if (tp.x.size() == tp.y.size()) {
      training_pairs_indices.push_back(training_pairs.size());
      training_pairs.push_back(tp);
      if (tp.x.size() > max_len) max_len = tp.x.size();
      tok_count += tp.x.size();
    }
    else {
      std::cerr << "Error: input and output vectors are of different lengths." << std::endl;
    }
  }

  /// @note Experimental
  unsigned prune(unsigned feature_count_threshold)
  {
    AttributesPruner attr_pruner(feature_counts,feature_count_threshold);
    unsigned pruned_attributes = 0;
    for (unsigned n = 0; n < training_pairs.size(); ++n) {
      TranslatedCRFInputSequence& x = training_pairs[n].x;
      for (unsigned t = 0; t < x.size(); ++t) {
        AttributeIDVector& attributes = boost::get<1>(x[t]);
        unsigned a_cnt = attributes.size();
        attributes.erase(std::remove_if(attributes.begin(), attributes.end(), attr_pruner), attributes.end());
        AttributeIDVector(attributes).swap(attributes);
        pruned_attributes += (a_cnt - attributes.size());
      } // for t
    } // for n
    return pruned_attributes;
  }

  /// Return then number of input tokens in the corpus
  unsigned token_count()      const { return tok_count; }
  /// Return then number of different attributes in the corpus
  unsigned attributes_count() const { return attributes_mapper.size(); }
  /// Return then number of different labels in the corpus
  unsigned labels_count()     const { return labels_mapper.size(); }

  /** 
    @brief  Reduce the space requirements of the corpus.
    @note   may not be feasible since during compression two copies of the corpus
            will be hold within memory.
  */
  void compress()
  {
    //std::vector<TranslatedCRFTrainingPair>(training_pairs).swap(training_pairs);
    std::vector<unsigned>(training_pairs_indices).swap(training_pairs_indices);
    attributes_mapper.compress();
    labels_mapper.compress();
  }

  /// Randomly permute the training pairs
  void random_shuffle()
  {
    std::random_shuffle(training_pairs_indices.begin(), training_pairs_indices.end());
  }

  /// Return a reference to the attributes mapper (mapping attributes strings to attribute IDs)
  const StringUnsignedMapper& get_attributes_mapper() const 
  {
    return attributes_mapper;
  }

  /// Return a reference to the labels mapper (mapping label strings to label IDs)
  const StringUnsignedMapper& get_labels_mapper() const 
  {
    return labels_mapper;
  }

  void clear_string_mappers()
  {
    attributes_mapper.clear();
  }

private:
  struct AttributesPruner
  {
    AttributesPruner(const FeatureCountMap& fc, unsigned ft) 
    : feature_counts(fc), feature_counts_threshold(ft) {}

    inline bool operator()(unsigned a_id) const
    { 
      auto f = feature_counts.find(a_id);
      return (f != feature_counts.end() && f->second < feature_counts_threshold); 
    }

    const FeatureCountMap& feature_counts;
    unsigned feature_counts_threshold;
  }; // AttributesPruner


private:
  // Read a training corpus from a tab-separated file
  void read2(std::istream& corpus_in)
  {
    typedef boost::char_separator<char>           CharSeparator;
    typedef boost::tokenizer<CharSeparator>       Tokenizer;

    const CharSeparator segmenter("\t ");
    std::string line, token, label;
    std::vector<std::string> tokens;
    CRFInputSequence current_input_seq;
    LabelSequence current_label_seq;

    unsigned n_lines = 0, n_seq = 0;
    while (corpus_in.good()) {
      std::getline(corpus_in,line);
      ++n_lines;
      if (line.empty()) {
        if (!current_input_seq.empty()) {
          add(CRFTrainingPair(current_input_seq,current_label_seq));
          current_input_seq.clear();
          current_label_seq.clear();
          ++n_seq;
        }
      }
      else {
        // Get rid of ^M at the end of Windows CRLF lines
        if (*line.rbegin() == 13) {
          line.resize(line.size()-1);
        }
        Tokenizer tokenizer(line,segmenter);
        auto tok_iter = tokenizer.begin();
        if (tok_iter == tokenizer.end()) { 
          std::cerr << "Invalid line: " << line << std::endl; continue; 
        }
        token = *tok_iter; tok_iter++;
        if (tok_iter == tokenizer.end()) { 
          std::cerr <<  "Invalid line: " << line << std::endl; continue; 
        }
        label = *tok_iter; tok_iter++;
        current_label_seq.push_back(label);
        current_input_seq.push_back(WordWithAttributes(token,AttributeVector(tok_iter,tokenizer.end())));
      }
      if ((n_lines % 100000) == 0) std::cerr << ".";
    } // while

    compress();
  }

  void read(std::istream& corpus_in)
  {
    typedef boost::char_separator<char>           CharSeparator;
    typedef boost::tokenizer<CharSeparator>       Tokenizer;

    const CharSeparator segmenter("\t ");
    std::string line;
    std::vector<std::string> tokens;
    CRFInputSequence current_input_seq;
    LabelSequence current_label_seq;

    unsigned n_lines = 0, n_seq = 0;
    while (corpus_in.good()) {
      std::getline(corpus_in,line);
      ++n_lines;
      if (line.empty()) {
        if (!current_input_seq.empty()) {
          add(CRFTrainingPair(current_input_seq,current_label_seq));
          current_input_seq.clear();
          current_label_seq.clear();
          ++n_seq;
        }
      }
      else {
        // Get rid of ^M at the end of Windows CRLF lines
        if (*line.rbegin() == 13) {
          line.resize(line.size()-1);
        }
        Tokenizer tokenizer(line,segmenter);
        tokens.assign(tokenizer.begin(),tokenizer.end());
        if (tokens.size() >= 2) {
          current_label_seq.push_back(tokens[1]);
          current_input_seq.push_back(
            WordWithAttributes(tokens[0],AttributeVector(tokens.begin()+2,tokens.end())));
        }
        else std::cerr << "Invalid line: " << line << std::endl;
      }
      if ((n_lines % 100000) == 0) std::cerr << ".";
    } // while

    compress();
  }

  /// Maps label
  inline LabelID map_label(const Label& l) 
  {
    LabelID l_id = labels_mapper.get_id(l);
    if (l_id == LabelID(-1)) {
      labels_mapper.add_pair(l,label_counter);
      l_id = label_counter;
      ++label_counter;
    }
    return l_id;
  }

  inline AttributeID map_attr(const Attribute& a)
  {
    AttributeID a_id = attributes_mapper.get_id(a);
    if (a_id == AttributeID(-1)) {
      attributes_mapper.add_pair(a,attr_counter);
      a_id = attr_counter;
      ++attr_counter;
      ++feature_counts[a_id];
    }
    return a_id;
  }

  inline unsigned map_token(const std::string& tok)
  {
    unsigned t_id = token_mapper.get_id(tok);
    if (t_id == unsigned(-1)) {
      token_mapper.add_pair(tok,token_type_counter);
      t_id = attr_counter;
      ++token_type_counter;
    }
    return t_id;
  }

  void register_token(const std::string& tok)
  {
  }

private:
  std::vector<TranslatedCRFTrainingPair>    training_pairs;
  std::vector<unsigned>                     training_pairs_indices;
  std::set<std::string>                     all_tokens;
  StringUnsignedMapper                      attributes_mapper;
  StringUnsignedMapper                      labels_mapper;
  StringUnsignedMapper                      token_mapper;
  FeatureCountMap                           feature_counts;
  unsigned                                  max_len;
  unsigned                                  tok_count;
  unsigned                                  attr_counter;
  unsigned                                  label_counter;
  unsigned                                  token_type_counter;
}; // TrainingCorpus

#endif

/*
/// CRFTrainingCorpus represents an annotated corpus
class CRFTrainingCorpus
{
public:
  typedef SimpleLinearCRFModel::Label                 Label;
  typedef SimpleLinearCRFModel::Attribute             Attribute;
  typedef CRFDecoder::CRFInputSequence                CRFInputSequence;
  typedef CRFDecoder::LabelSequence                   LabelSequence;
  typedef CRFDecoder::AttributeVector                 AttributeVector;
  typedef CRFDecoder::WordWithAttributes              WordWithAttributes;
  typedef std::pair<CRFInputSequence,LabelSequence>   CRFTrainingPair;
  
public:
  CRFTrainingCorpus() : max_len(0) {}
  
  unsigned size() const { return training_pairs.size(); }

  unsigned max_input_length() const { return max_len; }

  const CRFTrainingPair& operator[](unsigned index) const
  {
    static CRFTrainingPair invalid;
    return (index < size()) ? training_pairs[index] : invalid;
  }

  // Add a pair to the corpus
  void add(const CRFTrainingPair& tp) 
  {
    if (tp.first.size() == tp.second.size()) {
      training_pairs.push_back(tp);
      if (tp.first.size() > max_len) 
        max_len = tp.first.size();
      corpus_labels.insert(tp.second.begin(),tp.second.end());
      for (unsigned i = 0; i < tp.first.size(); ++i) {
        const AttributeVector& attrs = tp.first[i].attributes;
        corpus_attributes.insert(attrs.begin(),attrs.end());
      }
    }
    else {
      std::cerr << "Error: input annd output vectors are of different lengths." << std::endl;
    }
  }

  unsigned labels_count() const { return corpus_labels.size(); }
  unsigned attributes_count() const { return corpus_attributes.size(); }

  void compress()
  {
    std::vector<CRFTrainingPair>(training_pairs).swap(training_pairs);
  }

  void clear()
  {
    std::vector<CRFTrainingPair>().swap(training_pairs);
    corpus_labels.clear();
    corpus_attributes.clear();
  }

private:
  std::vector<CRFTrainingPair>    training_pairs;
  std::set<Label>                 corpus_labels;
  std::set<Attribute>             corpus_attributes;
  unsigned max_len;
}; // TrainingCorpus

*/
