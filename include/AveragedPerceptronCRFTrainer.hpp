////////////////////////////////////////////////////////////////////////////////////////////////////
// AveragedPerceptronCRFTrainer.hpp
// Definition of a class implementing an optimised version of the averaged perceptron algorithm
// by Collins 2002
// Thomas Hanneforth, Universität Potsdam
// March 2015
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __AVERAGEDPERCEPTRONCRFTRAINER_HPP__
#define __AVERAGEDPERCEPTRONCRFTRAINER_HPP__

#include "CRFTypedefs.hpp"
#include "CRFTraining.hpp"

#define PERCEPTRON_AMPLIFY_VALUE          0.2
#define PERCEPTRON_DAMPING_VALUE          -PERCEPTRON_AMPLIFY_VALUE

#define PERCEPTRON_TRANSITION_MULTIPLIER  2.0

/// Implements the averaged perceptron training algorithm
template<unsigned ORDER>
class AveragedPerceptronCRFTrainer : public CRFTrainer<ORDER>
{
public:
  /// Defines a state in a higher-order CRF
  typedef typename SimpleLinearCRFModel<ORDER>::CRFHigherOrderState  CRFHigherOrderState;

private:
  /**
    @brief ParamUpdater: updater function object for model parameters
      Updates both model parameter and model parameter sum
      The idea of the averaged parameter updater is the following:
      The perceptron algorithm is a sequential one, that is, the parameters are updated each time
      after processing a training pair based on the difference between the output sequence
      in the corpus and the one predicted by the model on the basis of the current parameter values.
      After that update, the whole parameter vector is added to a vector holding the parameter sum.
      This sum vector is responsible for the averaging step (some kind of smoothing step) : 
      parameters that are tied to output labels that stay correct during most of the training time 
      are downweighted less than parameters which cause wrong output labels most of the training time.
      Now, adding vectors of several hundred thousand or even million parameters after each prediction
      adds a big constant to the algorithm's time complexity. This is even more annoying since a single
      training pair affects only a small fraction of the parameters, the others stay the same.
      We solve that problem here by using two auxiliary vectors keeping track of the parameter updates.
      One vector (last_param_update) holds the time step when a given parameter was updated last,
      the other (last_model_params) holds the (non-averaged) value of that parameter at that time step.
      When the parameter is updated the next time (maybe several time steps later), all the parameter
      summations which were omitted in the mean time are now applied (based on the value in 
      last_model_params).
      In effect, repeated summations are replaced by rare multiplications.
      A last update of all parameters after training ensures that all pending summations are carried out.
    */
    struct ParamUpdater { 
    ParamUpdater(ParameterVector& params, ParameterVector& summed_params, 
                 ParameterVector& last_params, std::vector<unsigned>& last_update) 
    : model_params(params), summed_model_params(summed_params),
      last_model_params(last_params), last_param_update(last_update)
    {}
    
    /// Update parameter p with weight w at time step u
    inline void operator()(ParameterIndex p, unsigned u, Weight w) const 
    { 
      //if (p == ParameterIndex(-1)) return;
      // Add the weight to parameter p
      model_params[p] += w;
      if (u == last_param_update[p]) {
        summed_model_params[p] += w;
      }
      else {
        // n is the number of time steps since the last update of p
        unsigned n = u - last_param_update[p]-1;
        // Now perform the omitted summations
        summed_model_params[p] += model_params[p] + (n * last_model_params[p]);
        last_param_update[p] = u;
      }
      last_model_params[p] = model_params[p];
    }

    ParameterVector& model_params;              ///< The current parameters of the model
    ParameterVector& summed_model_params;       ///< The sum of the current model parameters
    ParameterVector& last_model_params;         ///< The value of the model parameters at the last update step
    std::vector<unsigned>& last_param_update;   ///< The time step of the last update
  }; // ParamUpdater

  /// This is an updater object for the non-averaged perceptron algorithm
  /// This leads to much worser parameter values.
  struct NonAveragedParamUpdater 
  { 
    NonAveragedParamUpdater(ParameterVector& params) : model_params(params) {}    
    inline void operator()(ParameterIndex p, unsigned u, Weight w) const { model_params[p] += w; }
    ParameterVector& model_params;
  }; // NonAveragedParamUpdater

public:
  /// Constructor: takes a translated training corpus
  AveragedPerceptronCRFTrainer(CRFTranslatedTrainingCorpus& training_corpus, unsigned pt=0)
  : CRFTrainer<ORDER>(training_corpus.get_labels_mapper(),training_corpus.get_attributes_mapper()),
    crf_decoder(CRFTrainer<ORDER>::get_model()), translated_training_corpus(training_corpus)
  {
    // Translate attributes and labels of the corpus
    this->create_initial_model(training_corpus);
    crf_decoder.resize_matrices(training_corpus.max_input_length());
  }

  /// Perform the perceptron training with a given number of iterations
  void train_by_number_of_iterations(unsigned num_iterations)
  {
    train(num_iterations,0.0,false);
  }

  /// Perform the perceptron training with a threshold argument
  void train_by_threshold(float threshold)
  {
    train(10000,threshold,true);
  }

private:
  /// Train by number of iterations or threshold
  void train(unsigned num_iterations, float threshold, bool use_threshold)
  {
    std::cerr << "Estimating model parameters (" << num_iterations << " iterations)" << std::endl;

    // Create parameter updater
    ParameterVector& model_params = this->crf_model.get_parameters();
    ParameterVector summed_model_params(this->crf_model.parameters_count(),Weight(0.0));
    ParameterVector last_params(this->crf_model.parameters_count(),Weight(0.0));
    std::vector<unsigned> last_update(this->crf_model.parameters_count(),0);
    ParamUpdater param_updater(model_params,summed_model_params,last_params,last_update);

    // z will hold the predicted output sequence
    LabelIDSequence z(translated_training_corpus.max_input_length());

    unsigned time_step = 0;
    for (unsigned t = 0; t < num_iterations; ++t) {
      time_t iter_start = clock();
      float loss = 0;
      // Iterate over the training instances
      for (unsigned i = 0; i < translated_training_corpus.size(); ++i) {
        const TranslatedCRFTrainingPair& x_y = translated_training_corpus[i];
        z.resize(x_y.x.size(),0);
        // Determine the currently best sequence for x
        crf_decoder.best_sequence(x_y.x,z);
        // Compare the two sequences
        unsigned num_diffs = 0;
        // Parameter updates are only necessary in case corpus and predicted output sequence differ
        if (x_y.y != z) {
          if (ORDER == 1) num_diffs = first_order_updater(x_y.x,x_y.y,z,param_updater,time_step);
          else            num_diffs = higher_order_updater(x_y.x,x_y.y,z,param_updater,time_step);
        }
        ++time_step;
        loss += num_diffs / float(x_y.y.size());
      } // for i

      std::cerr << "Iteration " << t+1 << ": loss: " << loss
                << ", time: " << ((clock() - iter_start)/float(CLOCKS_PER_SEC))  << "s"  << std::endl;

      // Permute the training corpus
      translated_training_corpus.random_shuffle();
      if (use_threshold && loss <= threshold) 
        break;
    } // for t

    // Now perform the pending parameter updates and divide all parameter values by Num-Iterations * |Corpus|
    average_parameters(summed_model_params,model_params,last_params,last_update, 
                       translated_training_corpus.size() * num_iterations);

    /// Write the averaged parameters back to the model
    this->crf_model.set_parameters(summed_model_params);
  }

  /// Update parameters for first-order CRFs
  unsigned first_order_updater(const TranslatedCRFInputSequence& x, 
                               const LabelIDSequence& y, const LabelIDSequence& z,
                               ParamUpdater& param_updater, unsigned time_step) const
  {
    unsigned num_diffs = 0;
    ParameterIndex p_ty, p_tz;
    LabelID prev_y = LabelID(-1), prev_z = LabelID(-1);
    for (unsigned j = 0; j < y.size(); ++j) {
      if (y[j] != z[j]) {
        // Differing label => update parameters for state features at the differing labels
        // and also the transitions leading to them
        compute_state_features(param_updater,x[j],y[j],time_step,PERCEPTRON_AMPLIFY_VALUE);
        compute_state_features(param_updater,x[j],z[j],time_step,PERCEPTRON_DAMPING_VALUE);

        // Handle transitions
        if (j > 0) {
          p_ty = this->crf_model.transition_param_index(prev_y,y[j]);
          if (p_ty != ParameterIndex(-1))
            param_updater(p_ty,time_step,PERCEPTRON_AMPLIFY_VALUE*PERCEPTRON_TRANSITION_MULTIPLIER);
          p_tz = this->crf_model.transition_param_index(prev_z,z[j]);
          if (p_tz != ParameterIndex(-1))
            param_updater(p_tz,time_step,PERCEPTRON_DAMPING_VALUE*PERCEPTRON_TRANSITION_MULTIPLIER);
        }
        ++num_diffs;
      }
      else if (prev_y != prev_z) {
        // Labels at current position are equal but previous labels differ =>
        // update transition parameters
        p_ty = this->crf_model.transition_param_index(prev_y,y[j]);
        if (p_ty != ParameterIndex(-1)) 
          param_updater(p_ty,time_step,PERCEPTRON_AMPLIFY_VALUE*PERCEPTRON_TRANSITION_MULTIPLIER);
        p_tz = this->crf_model.transition_param_index(prev_z,z[j]);
        if (p_tz != ParameterIndex(-1)) 
          param_updater(p_tz,time_step,PERCEPTRON_DAMPING_VALUE*PERCEPTRON_TRANSITION_MULTIPLIER);
      } // prev_z1 != prev_z2
      
      prev_y = y[j];
      prev_z = z[j];
    } // for j

    return num_diffs;
  }

  /// Update parameters for first-order CRFs
  unsigned higher_order_updater(const TranslatedCRFInputSequence& x, 
                                const LabelIDSequence& y, const LabelIDSequence& z,
                                ParamUpdater& param_updater, unsigned time_step) const
  {
    unsigned num_diffs = 0;
    int last_diff = -ORDER;
    CRFHigherOrderState from_y, to_y, from_z, to_z;
    for (int j = 0; j < y.size(); ++j) {
      if (y[j] != z[j]) {
        // Differing label => update parameters for state features
        compute_state_features(param_updater,x[j],y[j],time_step,PERCEPTRON_AMPLIFY_VALUE);
        compute_state_features(param_updater,x[j],z[j],time_step,PERCEPTRON_DAMPING_VALUE);      
        last_diff = j;
        ++num_diffs;
      }

      if (y[j] != z[j] || j < last_diff+ORDER) {
        // Either differing label or there is a label difference within a window of size ORDER
        // Update transition structure
        // 1) First create states with history ORDER (if possible)
        const LabelID* from_y_l = &y[0];
        const LabelID* from_z_l = &z[0];
        if ((j-int(ORDER)) >= 0) {
          from_y_l = &y[j-ORDER];
          from_z_l = &z[j-ORDER];
        }
        from_y.construct(from_y_l, &y[j]);
        from_z.construct(from_z_l, &z[j]);
        // 2) Update the transition parameters (there are not only the transitions with history length ORDER
        //    to consider, but also transitions of lower orders
        update_transition_parameters(param_updater,from_y,y[j],time_step,PERCEPTRON_AMPLIFY_VALUE);
        update_transition_parameters(param_updater,from_z,z[j],time_step,PERCEPTRON_DAMPING_VALUE);
      }
    } // for j
    return num_diffs;
  }

  /// 
  void compute_state_features(ParamUpdater& param_updater, const WordWithAttributeIDs& x, 
                              LabelID z, unsigned u, Weight uw) const
  {
    const AttributeIDVector& possible_true_attrs = boost::get<1>(x);
    // Iterate over the attributes of the current input pos
    for (auto a = possible_true_attrs.begin(); a != possible_true_attrs.end(); ++a) {
      ParameterIndex p_a = this->crf_model.get_param_index_for_attr_at_label(*a,z);
      //const LabelIDParameterIndexPairVector& l = this->crf_model.get_labels_for_attribute(*a);
      if (p_a != ParameterIndex(-1)) {
        param_updater(p_a,u,uw);
      }
    } // for k
  }

  /// Update transitions for a higher-order model
  void update_transition_parameters(ParamUpdater& param_updater, CRFHigherOrderState& from, LabelID c, 
                                    unsigned time_step, Weight uw) const
  {
    //std::string from_s = from.as_string(&this->crf_model.get_labels_mapper());
    CRFStateID from_id = this->crf_model.get_crf_state_id(from);

    if (from.history_length() < ORDER) {
#ifdef HOCRF_ADD_LOWER_ORDER_TRANSITIONS
      // Update model substructure: every suffix state of from of length l will
      // have two transitions: one to an appropriate state of length l+1 (increasing
      // the history, e.g. (with ORDER=3) (<BOS> LOC) --> (<BOS> LOC OTHER)) and one 
      // to an appropriate state of length l (wrapping the history, 
      // e.g. (<BOS> LOC) --> (LOC OTHER))
      unsigned hl = from.history_length();
      for (unsigned l = 0; l < hl; ++l) {
        CRFStateID to_id = this->crf_model.get_crf_state_id(from.increase_history(c));
        update_transition_param(param_updater,from_id,to_id,time_step,uw);
        to_id = this->crf_model.get_crf_state_id(from.wrap(c));
        update_transition_param(param_updater,from_id,to_id,time_step,uw);
        from.shorten_history();
        from_id = this->crf_model.get_crf_state_id(from);
      } // for l
#else
      CRFStateID to_id = this->crf_model.get_crf_state_id(from.increase_history(c));
      update_transition_param(param_updater,from_id,to_id,time_step,uw);
#endif
    }
    else {
      CRFHigherOrderState to(from.wrap(c));
      //std::string to_s = to.as_string(&crf_model.get_labels_mapper());
      CRFStateID to_id = this->crf_model.get_crf_state_id(to);
      // Add main transition with history length == ORDER
      // Note: transitions are reversed for higher-order models
      update_transition_param(param_updater,from_id,to_id,time_step,uw);

#ifdef HOCRF_ADD_LOWER_ORDER_TRANSITIONS
      // Add lower-order transitions
      for (unsigned o = 0; o < ORDER-1; ++o) {
        from.shorten_history();
        // transition lower -> higher
        CRFStateID shortend_from_id = this->crf_model.get_crf_state_id(from);
        update_transition_param(param_updater,shortend_from_id,to_id,time_step,uw);
        // transition lower -> lower
        to.shorten_history();
        CRFStateID shortend_to_id = this->crf_model.get_crf_state_id(to);
        update_transition_param(param_updater,shortend_from_id,shortend_to_id,time_step,uw);
        // transition higher -> lower
        update_transition_param(param_updater,from_id,shortend_to_id,time_step,uw);
        from_id = this->crf_model.get_crf_state_id(from);
      } // for o
#endif
    }
  }

  /// Basically, the same as above but restricted to single transition given as a pair of state IDs
  inline void update_transition_param(ParamUpdater& param_updater, CRFStateID from_id, CRFStateID to_id,
                                      unsigned time_step, Weight uw) const
  {
    ParameterIndex p_t = this->crf_model.transition_param_index(to_id,from_id);
    if (p_t != ParameterIndex(-1)) 
      param_updater(p_t,time_step,uw);
  }

  /// Perform pending updates and divide all parameters by d
  void average_parameters(ParameterVector& summed_model_params,
                          const ParameterVector& model_params,
                          const ParameterVector& last_model_params,
                          const std::vector<unsigned>& last_param_update, 
                          unsigned d) const
  {
    for (unsigned p = 0; p < summed_model_params.size(); ++p) {
      if (d != last_param_update[p]) {
        unsigned n = d - last_param_update[p]-1;
        summed_model_params[p] += (n * last_model_params[p]);
      }
      summed_model_params[p] /= d;
    }
  }

  void add_parameters(ParameterVector& summed_model_params, const ParameterVector& model_params)
  {
    for (unsigned p = 0; p < summed_model_params.size(); ++p) {
      summed_model_params[p] += model_params[p];
    }
  }

private: // Member variables
  CRFTranslatedTrainingCorpus&    translated_training_corpus; ///< The training corpus
  CRFDecoder<ORDER>               crf_decoder;                ///< The decoder for finding best output sequences
}; // AveragedPerceptronCRFTrainer

#endif
