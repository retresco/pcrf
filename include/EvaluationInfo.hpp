#ifndef __EVALUATION_INFO_HPP__
#define __EVALUATION_INFO_HPP__

#include <string>

/// Computes precision/recall/F1 score
struct EvaluationInfo
{
  EvaluationInfo(const std::string& zh = "") 
  : zero_hypothesis(zh), total_labels(0), correct_labels(0), 
    true_positive_labels(), true_negative_labels(), 
    false_positive_labels(), false_negative_labels() 
  {}

  float accuracy() const 
  {
    return correct_labels/float(total_labels);
  }

  float precision() const 
  {
    if (!zero_hypothesis.empty()) 
      return true_positive_labels/float(true_positive_labels + false_positive_labels);
    return 0.0;
  }

  float recall() const 
  {
    if (!zero_hypothesis.empty()) 
      return true_positive_labels/float(true_positive_labels + false_negative_labels);
    return 0.0;
  }

  float f1_score() const
  {
    return 0.0;
  }

  void operator()(const std::string& inferred, const std::string& gold_std) 
  {
    ++total_labels;
    if (zero_hypothesis.empty()) {
      if (inferred == gold_std) { 
        ++correct_labels; 
      }
    }
    else {
      if (gold_std == zero_hypothesis) {
        if (inferred == gold_std) {
          ++true_negative_labels;
          ++correct_labels;
        }
        else ++false_positive_labels;
      }
      else {
        if (inferred == gold_std) {
          ++true_positive_labels;
          ++correct_labels;
        }
        else ++false_negative_labels;
      }
    }
  }

  unsigned total_labels;
  unsigned correct_labels;
  unsigned true_positive_labels;
  unsigned true_negative_labels;
  unsigned false_positive_labels;
  unsigned false_negative_labels;
  
  const std::string zero_hypothesis;
}; // EvaluationInfo

#endif
