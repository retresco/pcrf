#ifndef __EVALUATION_INFO_HPP__
#define __EVALUATION_INFO_HPP__

#include <string>
#include <boost/unordered_map.hpp>

/// Computes precision/recall/F1 score
struct EvaluationInfo
{
  typedef boost::unordered_map<std::string,unsigned>  CountMap;

  EvaluationInfo(std::string dummy="") : total_labels(0), correct_labels(0) {}

  /// Returns the global accuracy
  float accuracy() const 
  {
    return correct_labels/float(total_labels);
  }

  /// Returns the label-wise precision
  float precision(const std::string& label) const 
  {
    auto tp = true_positive_labels.find(label);
    if (tp == true_positive_labels.end()) return 0.0;
    auto fp = false_positive_labels.find(label);
    unsigned false_pos = (fp != false_positive_labels.end()) ? fp->second : 0;
    return tp->second/float(tp->second+false_pos);
  }

  /// Returns the overall (micro-averaged) precision
  float precision() const 
  {
    unsigned true_pos = 0, false_pos = 0;
    for (auto tp = true_positive_labels.begin(); tp != true_positive_labels.end(); ++tp) {
      true_pos += tp->second;
    }
    for (auto fp = false_positive_labels.begin(); fp != false_positive_labels.end(); ++fp) {
      false_pos += fp->second;
    }
    return true_pos/float(true_pos+false_pos);
  }

  /// Returns the label-wise recall
  float recall(const std::string& label) const 
  {
    auto tp = true_positive_labels.find(label);
    if (tp == true_positive_labels.end()) return 0.0;
    auto fn = false_negative_labels.find(label);
    unsigned false_neg = (fn != false_negative_labels.end()) ? fn->second : 0;
    return tp->second/float(tp->second+false_neg);
  }

  float recall() const 
  {
    unsigned true_pos = 0, false_neg = 0;
    for (auto tp = true_positive_labels.begin(); tp != true_positive_labels.end(); ++tp) {
      true_pos += tp->second;
    }
    for (auto fn = false_negative_labels.begin(); fn != false_negative_labels.end(); ++fn) {
      false_neg += fn->second;
    }
    return true_pos/float(true_pos+false_neg);
  }

  float f1_score() const
  {
    return 0.0;
  }

  /// Computes label-wise precision/recall.
  void operator()(const std::string& inferred_label, const std::string& gold_label) 
  {
    ++total_labels;
    if (inferred_label == gold_label) { 
      ++correct_labels;
      ++true_positive_labels[gold_label];
    }
    else {
      ++false_negative_labels[gold_label];
      ++false_positive_labels[inferred_label];
    }
  }

  unsigned total_labels;
  unsigned correct_labels;
  CountMap true_positive_labels;
  CountMap true_negative_labels;
  CountMap false_positive_labels;
  CountMap false_negative_labels;
}; // EvaluationInfo

#endif
