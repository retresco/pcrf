#ifndef __CRF_DECODER_HPP__
#define __CRF_DECODER_HPP__

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <limits>
#include <iterator>
#include <algorithm>

#include "CRFTypedefs.hpp"
#include "SimpleLinearCRFModel.hpp"

#define MINIMUM_WEIGHT   Weight(-std::numeric_limits<Weight>::max())

/// CRFDecoder implements a decoder for first-order and higher-order linear CRFs
template<unsigned ORDER>
class CRFDecoder
{
public:
  /// Creates an instance of the decoder based on the given CRF model 'm'
  CRFDecoder(const SimpleLinearCRFModel<ORDER>& m) : crf_model(m) 
  { }

  /// Computes argmax output p(output|input)
  inline Weight best_sequence(const TranslatedCRFInputSequence& input, LabelIDSequence& output)
  {
    if (ORDER == 1) return first_order_best_sequence(input,output);
    else return higher_order_best_sequence(input,output);
  }

  /// Resizes trellis, backpointer matrix and all aux. matrixes to the specified sizes
  void resize_matrices(unsigned max_input_len)
  {
    trellis.resize(max_input_len, WeightVector(this->crf_model.states_count(),MINIMUM_WEIGHT));
    back_pointers.resize(max_input_len, BackPointers(this->crf_model.states_count(),0));
    precomputed_weights.resize(max_input_len);
    for (unsigned t = 0; t < trellis.size(); ++t) {
      precomputed_weights[t].resize(this->crf_model.labels_count(),Weight(0.0));
    }
  }

private:
  /// Computes argmax output p(output|input) for first-order CRFs
  inline Weight first_order_best_sequence(const TranslatedCRFInputSequence& input, LabelIDSequence& output)
  {
    prepare_matrices(input.size());
    precompute_weights(input);
    ViterbiScoreComputer viterbi_scorer(crf_model,input,trellis,precomputed_weights,back_pointers);
    return viterbi_scorer.delta(output);
  }

  /// Computes argmax output p(output|input) for higher-order CRFs
  inline Weight higher_order_best_sequence(const TranslatedCRFInputSequence& input, LabelIDSequence& output)
  {
    prepare_matrices(input.size());
    precompute_weights(input);
    HigherOrderViterbiScoreComputer viterbi_scorer(crf_model,input,trellis,precomputed_weights,back_pointers);
    return viterbi_scorer.delta(output);
  }

  /// Resize and initialise all matrices
  void prepare_matrices(unsigned n)
  {
    if (n > trellis.size()) {
      // Add additional rows
      trellis.resize(n,WeightVector(crf_model.states_count(),MINIMUM_WEIGHT));
      back_pointers.resize(n, BackPointers(crf_model.states_count(),0));
      precomputed_weights.resize(n,WeightVector(crf_model.labels_count(),MINIMUM_WEIGHT));
    }

    // Initialise trellis and backpointers
    for (unsigned i = 0; i < n; ++i) {
      WeightVector& trellis_i = trellis[i];
      BackPointers& bp_i = back_pointers[i];
      WeightVector& pw_i = precomputed_weights[i];
      std::fill(trellis_i.begin(),trellis_i.end(),MINIMUM_WEIGHT);
      std::fill(pw_i.begin(),pw_i.end(),Weight(0.0));
      std::fill(bp_i.begin(),bp_i.end(),0);
    }
  }

  /// Creates a T x L matrix of precomputed weights
  void precompute_weights(const TranslatedCRFInputSequence& input)
  {
    for (unsigned t = 0; t < input.size(); ++t) {
      WeightVector& precomputed_weights_at_t = precomputed_weights[t];
      std::fill(precomputed_weights_at_t.begin(),precomputed_weights_at_t.end(),Weight(0.0));
      const AttributeIDVector& token_attrs = boost::get<1>(input[t]);
      for (auto attr_k = token_attrs.begin(); attr_k != token_attrs.end(); ++attr_k) {
        const LabelIDParameterIndexPairVector& labels = crf_model.get_labels_for_attribute(*attr_k);
        for (auto l = labels.begin(); l != labels.end(); ++l) {
          precomputed_weights_at_t[l->first] += crf_model[l->second];
        } // for l
      } // for k
    } // for t
  }

private: // Types
  typedef std::vector<Weight>                                           WeightVector;
  typedef std::vector<WeightVector>                                     WeightMatrix;
  typedef typename SimpleLinearCRFModel<ORDER>::TransitionConstIterator TransitionIterator;
  typedef std::vector<int>                                              BackPointers;
  typedef std::vector<BackPointers>                                     BackPointerMatrix;

  /// WeightComputer is the base class of the classes ViterbiScoreComputer, 
  /// ForwardScoreComputer and BackwardScoreComputer
  struct WeightComputer
  {
    WeightComputer(const SimpleLinearCRFModel<ORDER>& m, const TranslatedCRFInputSequence& i, 
                   WeightMatrix& t, WeightMatrix& w)
    : crf_model(m), input(i), trellis(t), precomputed_weights(w)
    {}
  
    void print_trellis(std::ostream& o) const
    {
      for (unsigned t = 0; t < input.size(); ++t) o << "\t" << t;
      o << std::endl;
      for (unsigned qj = 0; qj < state_count(); ++qj) {
        o << qj;
        for (unsigned t = 0; t < input.size(); ++t) {
          o << "\t" << trellis[t][qj];
        }
        o << std::endl;
      }
    }

  protected:
    /// Return the score of the state features at state qj and position t
    inline Weight label_psi(LabelID qj, unsigned t) const
    {
      return precomputed_weights[t][qj];
    }
    
    inline Weight label_psi(LabelID qj, const WordWithAttributeIDs& x) const
    {
      Weight w(0.0);
      const AttributeIDVector& attrs_at_current_input_position = boost::get<1>(x);
      // The true features are not precomputed => iterate over the features of the current input pos
      for (unsigned k = 0; k < attrs_at_current_input_position.size(); ++k) {
        // attrs_at_current_label[k] = attribute id for k-th attribute of input_n
        ParameterIndex p_a = crf_model.get_param_index_for_attr_at_label(attrs_at_current_input_position[k],qj);
        if (p_a != ParameterIndex(-1)) {
          w += crf_model[p_a];
        }
      } // for k
      return w;
    }

    inline unsigned state_count() const { return crf_model.states_count(); }

  protected:
    const SimpleLinearCRFModel<ORDER>&  crf_model;
    const TranslatedCRFInputSequence    input;
    WeightMatrix&                       trellis;
    WeightMatrix&                       precomputed_weights;
  }; // WeightComputer

  /// ViterbiScoreComputer computes the best label sequence for a given input
  struct ViterbiScoreComputer : public WeightComputer
  {
    ViterbiScoreComputer(const SimpleLinearCRFModel<ORDER>& m, const TranslatedCRFInputSequence& i,
                         WeightMatrix& trellis, WeightMatrix& pre_w, BackPointerMatrix& bp) 
    : WeightComputer(m,i,trellis,pre_w), back_pointers(bp)
    {
      compute_forward_trellis();
      //print_trellis(std::cout);
    }
    
    /// Find the highest score in the last column of the trellis and reconstruct the best path
    /// by following the backpointer sequence
    Weight delta(LabelIDSequence& output)
    {
      if (this->input.empty()) return Weight(0.0);

      Weight score(MINIMUM_WEIGHT);
      int global_back_pointer = -1; 
      const WeightVector& last_column = this->trellis[this->input.size()-1];
      for (unsigned qi = 0; qi < this->state_count(); ++qi) {
        if (last_column[qi] > score) {
          score = last_column[qi];
          global_back_pointer = qi;
        }
      }

      // Construct output sequence by following the backpointer sequence
      extract_label_sequence(global_back_pointer,output);
      return score;
    }
  
  private:
    /// Compute the viterbi trellis for the current input sequence
    void compute_forward_trellis() 
    {
      const TranslatedCRFInputSequence& x = this->input;
      if (x.empty()) return;
      TransitionIterator tr;

      // Compute initial column (there are no transitions, only state features)
      WeightVector& column_zero = this->trellis[0];
      for (unsigned qj = 0; qj < this->state_count(); ++qj) {
        column_zero[qj] = this->label_psi(qj,0);
      }

      for (unsigned t = 1; t < x.size(); ++t) {
        const WeightVector& delta_prev_t = this->trellis[t-1];
        WeightVector& delta_t = this->trellis[t];
        BackPointers& back_pointers_at_t = this->back_pointers[t];
        // Iterate over all states in the current column
        for (unsigned qj = 0; qj < this->state_count(); ++qj) {
          Weight max_score(MINIMUM_WEIGHT);
          // Consider only incoming transitions to current state qj
          for (tr = this->crf_model.ingoing_transitions_of(qj); !tr.at_end(); ++tr) {
            LabelID qi = tr.from();
            Weight w = delta_prev_t[qi] + tr.weight();
            // Note that the maximisation below takes only the score of the transitions's
            // origin and the weight of the transitions into account; the value of qj's
            // state features are added later
            if (w > max_score) {
              max_score = w; 
              back_pointers_at_t[qj] = qi; 
            }
          } // for tr
          // Finally add the state features of qj
          delta_t[qj] = max_score + this->label_psi(qj,t);
        } // for qj
      } // for t
    }

    /// Extracts in reverse the best label sequence
    void extract_label_sequence(int bp,LabelIDSequence& output) const 
    {
      if (bp == -1) {
        std::fill(output.begin(),output.end(),0);
        return;
      }
      for (int k = output.size()-1; k >= 0; --k) {
        output[k] = bp;
        bp = this->back_pointers[k][bp];
      }
    }

  private:
    BackPointerMatrix& back_pointers;
  }; // ViterbiScoreComputer

  /// HigherOrderViterbiScoreComputer computes the best label sequence for a given input
  struct HigherOrderViterbiScoreComputer : public WeightComputer
  {
    HigherOrderViterbiScoreComputer(const SimpleLinearCRFModel<ORDER>& m, const TranslatedCRFInputSequence& i,
                                    WeightMatrix& trellis, WeightMatrix& pre_w, BackPointerMatrix& bp) 
    : WeightComputer(m,i,trellis,pre_w), back_pointers(bp)
    {
      compute_forward_trellis();
    }
    
    /// Find the highest score in the last column of the trellis and reconstruct the best path
    /// by following the backpointer sequence
    Weight delta(LabelIDSequence& output)
    {
      if (this->input.empty()) return Weight(0.0);

      Weight score(MINIMUM_WEIGHT);
      int global_back_pointer = -1; 
      const WeightVector& last_column = this->trellis[this->input.size()-1];
      for (unsigned qi = 0; qi < this->state_count(); ++qi) {
        if (last_column[qi] > score) {
          score = last_column[qi];
          global_back_pointer = qi;
        }
      }

      // Construct output sequence by following the backpointer sequence
      extract_label_sequence(global_back_pointer,output);
      return score;
    }
  
  private:
    /// Compute the viterbi trellis for the current input sequence
    void compute_forward_trellis() 
    {
      const TranslatedCRFInputSequence& x = this->input;
      if (x.empty()) return;
      TransitionIterator tr;

      WeightVector& trellis_at_zero = this->trellis[0];
      BackPointers& back_pointers_at_zero = this->back_pointers[0];
      for (tr = this->crf_model.outgoing_transitions_of(this->crf_model.start_state()); !tr.at_end(); ++tr) {
        trellis_at_zero[tr.to()] = tr.weight();
        back_pointers_at_zero[tr.to()] = this->crf_model.start_state();
      }

      for (unsigned t = 0; t < x.size()-1; ++t) {
        WeightVector& trellis_at_t = this->trellis[t];
        WeightVector& trellis_at_t_plus_one = this->trellis[t+1];
        BackPointers& back_pointers_at_t_plus_one = this->back_pointers[t+1];
        // Iterate over all states (exclude <BOS>)
        for (LabelID from = 1; from < this->state_count(); ++from) {
          Weight& trellis_at_t_and_from = trellis_at_t[from];
          if (trellis_at_t_and_from != MINIMUM_WEIGHT) {
            // State from is reachable
            // First add label features for from (this will not change the backpointers)
            trellis_at_t_and_from += this->label_psi(this->crf_model.get_crf_state(from).label_id(),t);
            // Consider only outgoing transitions of 'from'
            for (tr = this->crf_model.outgoing_transitions_of(from); !tr.at_end(); ++tr) {
              auto to = tr.to();
              Weight w = trellis_at_t_and_from + tr.weight();
              Weight& trellis_at_t_plus_one_and_to = trellis_at_t_plus_one[tr.to()];
              if (w > trellis_at_t_plus_one_and_to) {
                trellis_at_t_plus_one_and_to = w;
                back_pointers_at_t_plus_one[tr.to()] = from;
              }
            } // for tr
          } // if
        } // for from 
      } // for t

      // Add state features for the states in the last column
      WeightVector& trellis_last_column = this->trellis[x.size()-1];
      for (LabelID q = 1; q < this->state_count(); ++q) {
        if (trellis_last_column[q] != MINIMUM_WEIGHT) {
          trellis_last_column[q] += this->label_psi(this->crf_model.get_crf_state(q).label_id(),x.size()-1);
        }
      } // for q
    }

    /// Extracts in reverse the best label sequence
    void extract_label_sequence(int bp,LabelIDSequence& output) const 
    {
      if (bp == -1) {
        std::fill(output.begin(),output.end(),0);
        return;
      }
      for (int k = output.size()-1; k >= 0; --k) {
        // Extract label ID from the higher-order state (this is always the last component of the state tuple)
        output[k] = this->crf_model.get_crf_state(CRFStateID(bp)).label_id();
        bp = back_pointers[k][bp];
      }
    }

  private:
    BackPointerMatrix& back_pointers;
  }; // ViterbiScoreComputer

private:
  const SimpleLinearCRFModel<ORDER>&    crf_model;
  WeightMatrix                          trellis;
  WeightMatrix                          precomputed_weights;
  BackPointerMatrix                     back_pointers;
}; // CRFDecoder

#endif
