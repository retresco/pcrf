#ifndef __CRF_TRAINING_HPP__
#define __CRF_TRAINING_HPP__

#include <vector>

#include <boost/tuple/tuple.hpp>

#include "CRFTypedefs.hpp"
#include "SimpleLinearCRFModel.hpp"
#include "CRFTrainingCorpus.hpp"
#include "CRFDecoder.hpp"

//#define HOCRF_ADD_LOWER_ORDER_TRANSITIONS

/** 
  @brief CRFTrainer: Base class of all CRF training algorithms
  @param ORDER is the order of the trained model
*/
template<unsigned ORDER>
class CRFTrainer
{
public:
  typedef typename SimpleLinearCRFModel<ORDER>::CRFHigherOrderState  CRFHigherOrderState;

public:
  /// Returns a reference to the trained model
  const SimpleLinearCRFModel<ORDER>& get_model() const { return crf_model; }

protected:
  /**
    @brief Constructor
    @param l_map a mapper object mapping labels to label IDs, as created during adding translated
           training pairs to a corpus
    @param a_map a mapper object mapping attributes to attribute IDs
  */
  CRFTrainer(const StringUnsignedMapper& l_map, const StringUnsignedMapper& a_map, unsigned ft=5)
  : crf_model(l_map,a_map), feature_threshold(ft)
  {}

  /// Create the initial model after the factorisation f(y_{i-1},y_i) and f(y_i,x_i)
  void create_initial_model(const CRFTranslatedTrainingCorpus& training_corpus)
  {
    std::cerr << "Building initial model (order=" << ORDER << ") ...";

    if (ORDER == 1) {
      create_initial_first_order_model(training_corpus);
    }
    else {
      create_initial_higher_order_model(training_corpus);
    }

    crf_model.finalise();

    //std::ofstream out("initial.model");
    //out << crf_model;

    std::cerr << " done" << std::endl;
    std::cerr << "[#attributes: " << crf_model.attributes_count() 
              << ", #labels: " << crf_model.labels_count();
    if (ORDER > 1) {
      std::cerr << ", #states: " << crf_model.states_count();
    }
    std::cerr << ", #features: " << crf_model.features_count() 
              << ", #transitions: " << crf_model.transitions_count() 
              << ", #parameters: " << crf_model.parameters_count()
              << "]" << std::endl;
  }

private:
  /// Create an initial first-order model after the factorisation f(y_{i-1},y_i) and f(y_i,x_i)
  void create_initial_first_order_model(const CRFTranslatedTrainingCorpus& training_corpus)
  {
    for (unsigned n = 0; n < training_corpus.size(); ++n) {
      const TranslatedCRFTrainingPair& x_y = training_corpus[n];
      const TranslatedCRFInputSequence& x = x_y.x;
      const LabelIDSequence& y = x_y.y;
      LabelID prev_l_id = LabelID(-1);
      for (unsigned i = 0; i < x.size(); ++i) {
        LabelID l_id = y[i];
        if (i > 0) {
          // Add zero-weight transition between labels
          crf_model.add_transition(prev_l_id, l_id);
        }
        const AttributeIDVector& attributes = boost::get<1>(x[i]);
        for (unsigned a = 0; a < attributes.size(); ++a) {
          // Register attribute at current label (= create feature function)
          crf_model.add_attr_for_label(l_id,attributes[a]);
        } // for a
        prev_l_id = l_id;
      } // for i
    } // for n
  }

  /// Create the initial model after the factorisation f(y_{i-1},y_i) and f(y_i,x_i)
  void create_initial_higher_order_model(const CRFTranslatedTrainingCorpus& training_corpus)
  {
    for (unsigned n = 0; n < training_corpus.size(); ++n) {
      const TranslatedCRFTrainingPair& x_y = training_corpus[n];
      const TranslatedCRFInputSequence& x = x_y.x;
      const LabelIDSequence& y = x_y.y;
      
      // Always start with state <BOS>
      CRFHigherOrderState from(crf_model.get_bos_label_id());
      // Iterate over the sequence pair (x,y)
      for (unsigned i = 0; i < x.size(); ++i) {
        // Add attributes for label y[i], that is, construct state features
        const AttributeIDVector& attributes = boost::get<1>(x[i]);
        for (unsigned a = 0; a < attributes.size(); ++a) {
          // Register attribute at current label (= create feature function)
          crf_model.add_attr_for_label(y[i],attributes[a]);
        } // for a

        if (from.history_length() < ORDER) {
          // Order not yet reached (we are still at the beginning of the sequence)
          CRFHigherOrderState to(from.increase_history(y[i]));
#ifdef HOCRF_ADD_LOWER_ORDER_TRANSITIONS
          // Build model substructure: every suffix state of from of length l will
          // have two transitions: one to an appropriate state of length l+1 (increasing
          // the history, e.g. BOS LOC --> BOS LOC OTHER) and one to an appropriate state 
          // of length l (wrapping the history, e.g. BOS LOC --> LOC OTHER)
          unsigned hl = from.history_length();
          for (unsigned l = 0; l < hl; ++l) {
            crf_model.add_transition(from,from.increase_history(y[i]));
            crf_model.add_transition(from,from.wrap(y[i]));
            from.shorten_history();
          }
#else     // No lower-order transitions
          crf_model.add_transition(from,to);
#endif
          from = to;
        }
        else {
          // Full history reached
          // Shift history and extend with current label at the right
          CRFHigherOrderState to(from.wrap(y[i]));
          // transition higher -> higher
          crf_model.add_transition(from,to);

#ifdef HOCRF_ADD_LOWER_ORDER_TRANSITIONS
          CRFHigherOrderState to_copy(to);
          // Add lower order transitions triggered by the transition from --> to
          for (unsigned o = 0; o < ORDER-1; ++o) {
            CRFHigherOrderState from_copy(from);
            from.shorten_history();
            // transition lower -> higher
            crf_model.add_transition(from,to_copy);
            to_copy.shorten_history();
            // transition lower -> lower
            crf_model.add_transition(from,to_copy);
            // transition higher -> lower
            crf_model.add_transition(from_copy,to_copy);
          }
#endif
          from = to;
        }
      } // for i
    } // for n
  }

protected:
  SimpleLinearCRFModel<ORDER> crf_model;  ///< Holds the model after training
  unsigned feature_threshold;             ///<
}; // CRFTrainer

#endif
