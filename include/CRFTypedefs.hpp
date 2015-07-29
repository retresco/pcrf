////////////////////////////////////////////////////////////////////////////////////////////////////
// CRFTypedefs.hpp
// General types and data structures for the PCRF suite
// Thomas Hanneforth, Universität Potsdam
// March 2015
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __CRFTYPEDEFS_HPP__
#define __CRFTYPEDEFS_HPP__


#include <vector>
#include <string>
#include <map>
#include <set>
#include <iostream>

#include <boost/tuple/tuple.hpp>

/// Available CRF training algorithms
typedef enum { crfTrainAveragedPerceptron, crfTrainSGDL2 }                CRFTrainingAlgorithm;

/// Type of an attribute (=like a feature, but without output label component)
typedef std::string                                                       Attribute;
/// Type of a label
typedef std::string                                                       Label;
/// Type of a parameter weight
typedef double                                                            Weight;
/// Attributes are uniquely mapped to AttributeIDs
typedef unsigned                                                          AttributeID;
/// Labels are uniquely mapped to LabelIDs
typedef unsigned short                                                    LabelID;
/// Type of the index into the parameter vector
typedef unsigned                                                          ParameterIndex;
/// Used for higher order CRFs
typedef LabelID                                                           CRFStateID;

/// Set of label strings
typedef std::set<Label>                                                   LabelSet;
/// Vector of attribute strings (result of the annotation in CRFFeatureExtractor)
typedef std::vector<Attribute>                                            AttributeVector;
/// Output label sequence
typedef std::vector<Label>                                                LabelSequence;

/// Vector of attribute IDs (resulting from the translation of AttributeVectors)
typedef std::vector<AttributeID>                                          AttributeIDVector;
typedef std::vector<LabelID>                                              LabelIDSequence;
/// Vector of label ID sequences
typedef std::vector<LabelIDSequence>                                      LabelIDSequenceVector;
typedef std::vector<ParameterIndex>                                       ParameterIndexVector;
/// Parameter vector stored in a CRF model
typedef std::vector<Weight>                                               ParameterVector;
typedef std::map<Weight,std::string>                                      WeightStringMap;
typedef std::pair<LabelID,ParameterIndex>                                 LabelIDParameterIndexPair;
typedef std::pair<LabelID,Weight>                                         LabelIDWeightPair;
typedef std::vector<LabelIDParameterIndexPair>                            LabelIDParameterIndexPairVector;
/// Holds after decoding the inferred output label ID sequences and its score
typedef std::pair<LabelIDSequence,Weight>                                 BestScoredSequence;

/// WordWithAttributeIDs is the chief data structure of a translated input sequence x.
/// It consists of the token and its translated attributes  
typedef boost::tuple<unsigned,AttributeIDVector>                          WordWithAttributeIDs;

/// WordWithAttributes represents a string token, together with its (string) attributes
struct WordWithAttributes
{
  WordWithAttributes(const std::string& t, const AttributeVector& a)
  : token(t), attributes(a) {}

  /// Tab-separated output on a stream
  friend std::ostream& operator<<(std::ostream& o, const WordWithAttributes& wa)
  {
    if (WordWithAttributes::GetOutputTokenFlag()) o << wa.token << "\t";
    std::copy(wa.attributes.begin(),wa.attributes.end(),std::ostream_iterator<std::string>(o,"\t"));
    return o;
  }

  std::string token;
  AttributeVector attributes;

  static void SetOutputTokenFlag(bool v)  { OutputToken = v; }
  static bool GetOutputTokenFlag()        { return OutputToken; }

private:
  static bool OutputToken;
}; // WordWithAttributes

bool WordWithAttributes::OutputToken = false;


/// CRFInputSequence represents x, the (untranslated) input sequences of a CRF.
typedef std::vector<WordWithAttributes>                                   CRFInputSequence;

/// CRFTrainingPair represents (x,y), the (untranslated) training pair
typedef std::pair<CRFInputSequence,LabelSequence>                         CRFTrainingPair;

/// TranslatedCRFInputSequence represents x, the (translated) input sequences of a CRF.
typedef std::vector<WordWithAttributeIDs>                                 TranslatedCRFInputSequence;

/// CRFTrainingPair represents (x,y), the (translated) training pair
struct TranslatedCRFTrainingPair
{
  /// Default instance
  TranslatedCRFTrainingPair() {}
  /// Constructed an instance of a translated training pair
  TranslatedCRFTrainingPair(const TranslatedCRFInputSequence _x, const LabelIDSequence& _y)
  : x(_x), y(_y) {}

  void set_attributes_in_sequences(const AttributeIDVector& a)
  {
    attributes_in_sequences = a;
  }

  TranslatedCRFInputSequence x;               ///< Input sequence (all attributes translated to IDs)
  LabelIDSequence y;                          ///< Output sequence (labels translated to IDs)
  AttributeIDVector attributes_in_sequences;  ///< List of all attribute IDs in the sequence
}; // TranslatedCRFTrainingPair

#endif
