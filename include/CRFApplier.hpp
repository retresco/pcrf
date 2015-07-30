#ifndef __CRFAPPLIER_HPP__
#define __CRFAPPLIER_HPP__

/**
  \page PageApplier CRFApplier file formats
  \section PageApplierInput Input files
    CRFApplier either accepts running text or tab-separated text files, both must be
    UTF-8 encoded. 
    \subsection PageApplierInputRunning Running text
    The input text file must be UTF-8 encoded. The tokenizer used in PCRF (AsyncTokenizer)
    reads the input file linewise and extracts sequence by sequence from the input file. 
    Usually, a sequence is an (English) sentence. The tokenizer does heuristic sentence 
    segmentation based on punctuation and has also classifies the input token as word, 
    number, date, email and web addresses, abbreviation, punctuation etc.
    A line can contain several CRF input sequences (sentences). An input sequence can 
    never cross a line boundary.
        
    \subsection PageApplierTSV Tab-separated data
    The column layout of tab-separated data is determined by an instance
    of CRFConfiguration. Their must be at least a token column containing the input
    token. In evaluation mode, a column containing the Gold standard output label must
    also be present. An empty line serves as boundary between input sequences.

    \section PageApplierOutput Output files
    The output of CRFApplier is controlled by a user-definable function object which
    creates the desired model output. Currently, there are the following predefined
    output objects:
    - NEROneWordPerLineOutputter outputs tab-separated data in the following layout:
      OUTPUTLABEL INPUTTOKEN TOKENIZERCLASS POSITION 
      where 
      - TOKENIZERCLASS is an element in {WORD, NUMBER, PUNCT, DATE, HTML-Entity, 
        L_QUOTE, R_QUOTE, GENITIVE_SUFFIX, DASH, L_BRACKET, R_BRACKET, XML/HTML}
      - POSITION is a pair (OFFSET,LEN) where OFFSET is the byte offset of the token
        and LEN its byte length
    - JSONOutputter outputs results in JSON syntax (see http://json.org). Currently,
      JSON output is only supported for named entity recognition tasks.
      The main JSON key is here "entities", whose values is a list of named entity 
      sub structures with the keys "surface", "entity_type", "start" and "end".
      The keys "start" and "end" are numbers denoting a half open interval where "start"
      is the byte offset where the named entity starts and "end" is the byte offset
      of the first byte following the named entity.
    - MorphOutputter is useful for outputting morphology results. The labels of an
      output sequence are separated by a space; after a sequence follows a line break.  
*/

#include <iostream>
#include <fstream>

#include "SimpleLinearCRFModel.hpp"
#include "CRFDecoder.hpp"
#include "CRFFeatureExtractor.hpp"
#include "CRFConfiguration.hpp"
#include "AsyncTokenizer.hpp"
#include "TokenWithTag.hpp"
#include "EvaluationInfo.hpp"

/**
  @brief CRFApplier applies an CRF model to text files representing column data or running text.
  The application is controlled by an instance of CRFConfiguration which determines which 
  features are selected during application. The template ORDER argument gives the order of
  the model passed to the constructor of CRFApplier.
*/
template<unsigned ORDER>
class CRFApplier
{
public:
  /**
    @brief Constructor for the CRF applier object.
    @param m the model to be applied
    @param conf the CRF configuration determining which features are selected during application
    @param dl debug level
  */
  CRFApplier(const SimpleLinearCRFModel<ORDER>& m, const CRFConfiguration& conf, unsigned dl = 0) 
  : crf_model(m), crf_config(conf), crf_decoder(m), crf_fe(conf.features()),
    enhanced_annotation_scheme(conf.annotation_scheme()==nerBILOU), 
    order(1), debug_level(dl), token_count(0), seq_count(0) 
  {
    // Load binary lists (context clues, named entities etc.)
    //load_lists();
    crf_fe.set_context_window_size(crf_config.get_context_window_size());
    crf_fe.set_inner_word_ngrams(crf_config.get_inner_word_ngrams());
  }

  /** 
    @brief  Applies the CRF model to a text file (containg table data or running text).
            The file must contain the correct output label for each token in a sequence
    @param  seq UTF-8 encoded sequence containing TokenWithTag instances to which apply_to()
            will add the inferred output label.
    @param  outputter Function object which implements the output of the results in different
            formats. The outputter object is called every time a full sequence was annotated 
            by the model. The outputter must be derived from NEROutputterBase.
    @todo   This is more or less redundant code! 
  */
  template<typename OUTPUT_METHOD>
  void apply_to(TokenWithTagSequence& seq, OUTPUT_METHOD& outputter)
  {
    TranslatedCRFInputSequence translated_crf_iseq(seq.size());
    LabelIDSequence inferred_label_ids(seq.size(),0);
    LabelSequence inferred_labels(seq.size());

    // Add and translate features
    CRFInputSequence crf_iseq = crf_fe.add_features(seq);       
    translate(crf_iseq,translated_crf_iseq);

    // Decode the input
    crf_decoder.best_sequence(translated_crf_iseq, inferred_label_ids);

    // Add labels to input sequence
    for (unsigned i = 0; i < translated_crf_iseq.size(); ++i) {
      inferred_labels[i] = crf_model.get_label(inferred_label_ids[i]);
      seq[i].assign_label(inferred_labels[i]);
    } // for i

    // Hand over to outputter
    outputter(seq);
  }

    /** 
    @brief Applies the CRF model to a text file (containing table data or running text).
           The file must contain the correct output label for each token in a sequence
    @param text_in text stream opened on an UTF-8 encoded text file.
    @param running_text true if the input is running text, false if it is table (TSV) data
    @param outputter Function object which implements the output of the results in different
           formats. The outputter object is called every time a full sequence was annotated 
           by the model. The outputter must be derived from NEROutputterBase.
  */
  template<typename OUTPUT_METHOD>
  void apply_to(std::istream& text_in, OUTPUT_METHOD& outputter, bool running_text)
  {
    EvaluationInfo e;
    if (running_text)
      apply_to_running_text(text_in,outputter,false,e);
    else 
      apply_to_column_data(text_in,outputter,false,e);
  }
  /** 
    @brief Applies the CRF model to a text file (containing table data or running text)
           and evaluates it.
           The file must contain the correct output label for each token in a sequence
    @param text_in text stream opened on an UTF-8 encoded text file.
    @param running_text true if the input is running text, false if it is table (TSV) data
    @param outputter Function object which implements the output of the results in different
           formats (see apply_to()). 
  */
  template<typename OUTPUT_METHOD>
  EvaluationInfo evaluation_of(std::istream& text_in, OUTPUT_METHOD& outputter, bool running_text)
  {
    EvaluationInfo e(crf_config.get_default_label());
    if (running_text)
      (text_in,outputter,true,e);
    else 
      apply_to_column_data(text_in,outputter,true,e);
    return e;
  }

  /// Resets all counters to 0
  void reset()
  { 
    token_count = seq_count = 0;
  }

  /// Returns the number of processed input tokens  
  unsigned processed_tokens() const 
  { 
    return token_count;
  }

  /// Returns the number of processed input sequences
  unsigned processed_sequences() const 
  { 
    return seq_count;
  }

  /** 
    @brief Add DAWG for left contexts
    @param dawg_in binary stream opened on a binary DWAG file
  */
  void add_left_contexts(std::ifstream& dawg_in)
  {
    crf_fe.add_left_contexts(dawg_in);
  }

  /** 
    @brief Add DAWG for right contexts
    @param dawg_in binary stream opened on a binary DWAG file
  */
  void add_right_context_list(std::ifstream& dawg_in)
  {
    crf_fe.add_right_contexts(dawg_in);
  }

  /** 
    @brief Add DAWG for Wiki/gazeteer patterns
    @param dawg_in binary stream opened on a binary DWAG file
  */
  void add_patterns(std::ifstream& dawg_in)
  {
    crf_fe.add_patterns(dawg_in);
  }

  /** 
    @brief Add regular expression list
    @param list_in text stream with two colums (tab-separated):
           Col1 Feature Some arbitrary string acting as a feature value
           Col2 A regular expression (see boost::regex for the syntax) 
                to be applied to the current input token.
  */
  void add_word_regex_list(std::ifstream& list_in)
  {
    crf_fe.add_word_regex_list(list_in);
  }

private:
  /**
    @brief Apply the model to all sequences in 'text_in'
    @param text_in a text stream open on a UTF-8 encoded text file
    @param outputter An output object
    @param eval_mode if true, precision/recall/F1-score is computed 
    @param eval_info object for holding precision/recall/F1-score
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

    // Get a sequence from the text file and tokenize it
    while (tokenizer.tokenize(sentence)) {
      token_count += sentence.size();
      ++seq_count;

      if (debug_level == 1) {
        output_sequence(sentence,seq_count);
      }
      
      // Add string features to the tokens of the sequence
      CRFInputSequence seq = crf_fe.add_features(sentence);
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
    } // while
  }

  /**
    @brief Apply the model to the column data in the text stream 'data_in'
    @param data_in a text stream open on a UTF-8 encoded text file
    @param outputter An output object (see NEROutputterBase)
    @param eval_mode if true, precision/recall/F1-score is computed 
    @param eval_info object for holding precision/recall/F1-score
  */
  template<typename OUTPUT_METHOD>
  void apply_to_column_data(
          std::istream& data_in, OUTPUT_METHOD& outputter, 
          bool eval_mode, EvaluationInfo& eval_info) 
  {
    std::vector<std::string> tokens;
    std::string line;
    TokenWithTagSequence sequence;
    LabelSequence inferred_labels;

    // Get the column numbers of all columns in the input
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
        // If an empty line is found the current sequence is complete
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
        // Tokenize the current line
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

  /// @TODO Still useful????
  void apply_model(
          TokenWithTagSequence& sequence, 
          LabelSequence& inferred_labels, 
          bool eval_mode, EvaluationInfo& eval_info)
  {
    TranslatedCRFInputSequence translated_seq;
    LabelIDSequence inferred_label_ids;
    std::string label_assigned_by_model;

    // Add string features to the tokens of the sequence
    CRFInputSequence seq = crf_fe.add_features(sequence);
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

  void load_patterns()
  {
//    if ((crf_config.features() & AllNELists) != 0) {
//      std::string fn = crf_config.get_filename("NamedEntities");
//      if (!fn.empty()) {
//        std::ifstream list_in(fn.c_str(),std::ios::binary);
//        if (list_in) {
//          std::cerr << "Loading " << fn << "\n";
//          add_ne_list(list_in);
//        }
//        else std::cerr << "\nError: Unable to open named entities list '" << fn << "'" << std::endl;
//      }
//      else {
//        std::cerr << "Warning: 'AllNELists' specified, but no filename for 'NamedEntities' given\n";
//      }
//    }
  }

private:
  const SimpleLinearCRFModel<ORDER>&  crf_model;                    ///< The model to be appiled
  const CRFConfiguration&             crf_config;                   ///< Configuration 
  bool                                enhanced_annotation_scheme;   ///< BIO or BILOU 
  CRFFeatureExtractor                 crf_fe;                       ///< Feature annotator
  CRFDecoder<ORDER>                   crf_decoder;                  ///< Decoder for finding the best output seq.
  unsigned                            token_count;                  ///< Number of tokens found
  unsigned                            seq_count;                    ///< Number of sequences found
  unsigned                            debug_level;
  unsigned                            order;                        ///< No longer used
}; // CRFApplier

#endif
