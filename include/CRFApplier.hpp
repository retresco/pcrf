#ifndef __CRFAPPLIER_HPP__
#define __CRFAPPLIER_HPP__

#include <iostream>
#include <fstream>

#include "SimpleLinearCRFModel.hpp"
#include "CRFDecoder.hpp"
#include "NERFeatureExtractor.hpp"
#include "NERConfiguration.hpp"
#include "AsyncTokenizer.hpp"
#include "TokenWithTag.hpp"
#include "EvaluationInfo.hpp"


template<unsigned ORDER>
class CRFApplier
{
public:
  /**
    @brief Constructor
    @param m the model to be applied
    @conf
    @param dl debug level
  */
  CRFApplier(const SimpleLinearCRFModel<ORDER>& m, const NERConfiguration& conf, unsigned dl = 0) 
  : crf_model(m), crf_config(conf), crf_decoder(m), ner_fe(conf.features()),
    enhanced_annotation_scheme(conf.annotation_scheme()==nerBILOU), 
    order(1), debug_level(dl), token_count(0), seq_count(0) 
  {
    // Load binary lists (context clues, named entities etc.)
    //load_lists();
    ner_fe.set_context_window_size(crf_config.get_context_window_size());
  }

  template<typename OUTPUT_METHOD>
  void apply_to(std::istream& text_in, OUTPUT_METHOD& outputter)
  {
    EvaluationInfo e;
    if (crf_config.input_is_running_text())
      apply_to_running_text(text_in,outputter,false,e);
    else 
      apply_to_column_data(text_in,outputter,false,e);
  }

  template<typename OUTPUT_METHOD>
  EvaluationInfo evaluation_of(std::istream& text_in, OUTPUT_METHOD& outputter)
  {
    EvaluationInfo e(crf_config.get_default_label());
    if (crf_config.input_is_running_text())
      apply_to_running_text(text_in,outputter,true,e);
    else 
      apply_to_column_data(text_in,outputter,true,e);
    return e;
  }

  void reset()
  { 
    token_count = seq_count = 0;
  }

  unsigned processed_tokens() const 
  { 
    return token_count;
  }

  unsigned processed_sequences() const 
  { 
    return seq_count;
  }

  void add_left_context_list(std::ifstream& list_in)
  {
    ner_fe.add_left_context_list(list_in);
  }

  void add_right_context_list(std::ifstream& list_in)
  {
    ner_fe.add_right_context_list(list_in);
  }

  void add_ne_list(std::ifstream& list_in)
  {
    ner_fe.add_ne_list(list_in);
  }

  void add_person_names_list(std::ifstream& list_in)
  {
    ner_fe.add_person_names_list(list_in);
  }

  void add_word_regex_list(std::ifstream& list_in)
  {
    ner_fe.add_word_regex_list(list_in);
  }

private:
  /**
    @brief Apply the model to all sequences in 'text_in'
    @param text_in
    @param outputter
  */
  template<typename OUTPUT_METHOD>
  void apply_to_running_text(
          std::istream& text_in, OUTPUT_METHOD& outputter, 
          bool eval_mode, EvaluationInfo& eval_info) 
  {
    TokenWithTagSequence sentence;
    TranslatedCRFInputSequence translated_seq;
    LabelIDSequence inferred_label_ids;
    LabelSequence inferred_labels;
    AsyncTokenizer tokenizer(text_in,enhanced_annotation_scheme,order,crf_config.get_default_label());
    std::string label_assigned_by_model;

    bool done = tokenizer.tokenize(sentence);
    token_count += sentence.size();

    while (!done) {
      if (debug_level == 1) {
        output_sequence(sentence,seq_count);
      }
      
      // Add string features to the tokens of the sequence
      CRFInputSequence seq = ner_fe.add_features(sentence);
      if (debug_level == 1) {
        std::copy(seq.begin(),seq.end(),std::ostream_iterator<WordWithAttributes>(std::cout,"\n"));
        std::cout << std::endl;
      }

      // Translate the features to feature ids
      translate(seq,translated_seq);

      // Decode the input
      inferred_label_ids.resize(translated_seq.size(),0);
      inferred_labels.resize(translated_seq.size());
      crf_decoder.best_sequence(translated_seq, inferred_label_ids);

      // Add labels to input sentence
      for (unsigned i = 0; i < translated_seq.size(); ++i) {
        inferred_labels[i] = crf_model.get_label(inferred_label_ids[i]);
        if (eval_mode) {
          eval_info(inferred_labels[i],sentence[i].label);
        }
        else {
          sentence[i].assign_label(inferred_labels[i]);
        }
      } // for i

      // Hand over to outputter
      if (eval_mode) {
        outputter(sentence,inferred_labels);
      }
      else {
        outputter(sentence);
      }

      // Start over
      sentence.clear();
      ++seq_count;
      done = tokenizer.tokenize(sentence);
      token_count += sentence.size();
    } // while
  }

  template<typename OUTPUT_METHOD>
  void apply_to_column_data(
          std::istream& data_in, OUTPUT_METHOD& outputter, 
          bool eval_mode, EvaluationInfo& eval_info) 
  {
    std::vector<std::string> tokens;
    std::string line;
    TokenWithTagSequence sequence;
    LabelSequence inferred_labels;

    unsigned col_count = crf_config.columns_count();
    unsigned token_column = crf_config.get_column_no("Token");
    unsigned label_column = crf_config.get_column_no("Label");
    unsigned tag_column = crf_config.get_column_no("Tag");
    unsigned position_column = crf_config.get_column_no("Position");
    unsigned lemma_column = crf_config.get_column_no("Lemma");

    if (token_column == unsigned(-1)) {
      std::cerr << "Missing token column\n";
    }

    if (eval_mode && label_column == unsigned(-1)) {
      std::cerr << "Missing label column, but evaluation mode specified\n";
    }

    while (data_in.good()) {
      std::getline(data_in,line);
      if (line.empty()) {
        if (!sequence.empty()) {
          /// Apply the model to the sequence
          apply_model(sequence,inferred_labels,eval_mode,eval_info);
          // Hand over to outputter
          if (eval_mode) {
            outputter(sequence,inferred_labels);
          }
          else {
            outputter(sequence);
          }
          ++seq_count;
          token_count += sequence.size();
          sequence.clear();
        }
      }
      else {
        boost::tokenizer<boost::char_separator<char>  > tokenizer(line, boost::char_separator<char>("\t "));
        tokens.assign(tokenizer.begin(),tokenizer.end());
        if (tokens.size() == col_count) {
          TokenWithTag tt(tokens[token_column]);
          tt.assign_label(tokens[label_column]);
          if (tag_column != unsigned(-1)) 
            tt.assign_tag(tokens[tag_column]);
          sequence.push_back(tt);
        }
      }
    } // while
    // TODO: process last sequence
  }

  void apply_model(
          TokenWithTagSequence& sequence, 
          LabelSequence& inferred_labels, 
          bool eval_mode, EvaluationInfo& eval_info)
  {
    TranslatedCRFInputSequence translated_seq;
    LabelIDSequence inferred_label_ids;
    std::string label_assigned_by_model;

    // Add string features to the tokens of the sequence
    CRFInputSequence seq = ner_fe.add_features(sequence);
    if (debug_level == 1) {
      std::copy(seq.begin(),seq.end(),std::ostream_iterator<WordWithAttributes>(std::cout,"\n"));
      std::cout << std::endl;
    }

    // Translate the features to feature ids
    translate(seq,translated_seq);

    // Decode the input
    inferred_label_ids.resize(translated_seq.size(),0);
    inferred_labels.resize(translated_seq.size());
    crf_decoder.best_sequence(translated_seq, inferred_label_ids);

    // Add labels to input sentence
    for (unsigned i = 0; i < translated_seq.size(); ++i) {
      inferred_labels[i] = crf_model.get_label(inferred_label_ids[i]);
      if (eval_mode) {
        eval_info(inferred_labels[i],sequence[i].label);
      }
      else {
        sequence[i].assign_label(inferred_labels[i]);
      }
    } // for i
  }

  void translate(CRFInputSequence& seq, TranslatedCRFInputSequence& translated_seq)
  {
    AttributeIDVector a_ids; 
    ParameterIndexVector p;
    translated_seq.clear();
    for (unsigned i = 0; i < seq.size(); ++i) {
      const WordWithAttributes& w = seq[i];
      a_ids.clear();
      for (unsigned a = 0; a < w.attributes.size(); ++a) {
        AttributeID a_id = crf_model.get_attr_id(w.attributes[a]);
        if (a_id != AttributeID(-1)) {
          a_ids.push_back(a_id);
        }
      }
      //translated_seq.push_back(WordWithAttributeIDs(w.token,a_ids));
      // Note: the CRF decoder doesnt look at the actual tokens, so we simply add a dummy ID 0.
      translated_seq.push_back(WordWithAttributeIDs(0,a_ids));
    }
  }

  void output_sequence(const TokenWithTagSequence& sentence, unsigned seq_count) const
  {
    std::cout << "Sentence # " << (seq_count+1) << std::endl;
    std::copy(sentence.begin(),sentence.end(),std::ostream_iterator<TokenWithTag>(std::cout,"\n"));
  }

  void load_lists()
  {
    if ((crf_config.features() & AllNELists) != 0) {
      std::string fn = crf_config.get_filename("NamedEntities");
      if (!fn.empty()) {
        std::ifstream list_in(fn.c_str(),std::ios::binary);
        if (list_in) {
          std::cerr << "Loading " << fn << "\n";
          add_ne_list(list_in);
        }
        else std::cerr << "\nError: Unable to open named entities list '" << fn << "'" << std::endl;
      }
      else {
        std::cerr << "Warning: 'AllNELists' specified, but no filename for 'NamedEntities' given\n";
      }
    }
  }

private:
  const SimpleLinearCRFModel<ORDER>&  crf_model;                    ///< The model to be appiled
  const NERConfiguration&             crf_config;                   ///< Configuration 
  bool                                enhanced_annotation_scheme;   ///< BIO or BILOU 
  NERFeatureExtractor                 ner_fe;                       ///< Feature annotator
  CRFDecoder<ORDER>                   crf_decoder;                  ///< Decoder for finding the best output seq.
  unsigned                            order;
  unsigned                            token_count;
  unsigned                            seq_count;
  unsigned                            debug_level;
}; // CRFApplier

#endif
