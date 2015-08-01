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
    if (fp == false_positive_labels.end()) return std::numeric_limits<float>::max();
    return tp->second/float(tp->second+fp->second);
  }

  float precision() const 
  {
    return 0.0;
  }

  /// Returns the label-wise recall
  float recall(const std::string& label) const 
  {
    auto tp = true_positive_labels.find(label);
    if (tp == true_positive_labels.end()) return 0.0;
    auto fn = false_negative_labels.find(label);
    if (fn == false_negative_labels.end()) return std::numeric_limits<float>::max();
    return tp->second/float(tp->second+fn->second);
  }

  float recall() const 
  {
    return 0.0;
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
