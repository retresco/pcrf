////////////////////////////////////////////////////////////////////////////////////////////////////
// SimpleLinearCRFModel.hpp
// Definition of a class for representing first or higher-order Conditional Random Fields
// Thomas Hanneforth, Universität Potsdam
// March 2015
// TODO: 
//  * Text/binary output of hoCRFs
//  * checksums for read_model()
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __SIMPLE_LINEAR_CRF_MODEL_HPP__
#define __SIMPLE_LINEAR_CRF_MODEL_HPP__

// Todo: consider replacing attributes map by a qcdb

#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <new>

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/unordered_map.hpp>

#include "CRFTypedefs.hpp"
#include "StringUnsignedMapper.hpp"


#define MODEL_HEADER_ID   "LCRF Binary Model File version 1.0"

#define BOS_LABEL         0

/// Metadata of a simple linear CRF model
struct SimpleLinearCRFModelMetaData
{
  unsigned order;                         ///<
  unsigned num_labels;                    ///<
  unsigned num_states;                    ///<
  unsigned num_transitions;               ///<
  unsigned num_attributes;                ///<
  unsigned num_features;                  ///<
  unsigned num_parameters;                ///<
  unsigned num_non_null_parameters;       ///<
}; // SimpleLinearCRFModelMetaData


/// SimpleLinearCRFModel implements a simple linear CRF of order ORDER
template<unsigned ORDER=1>
class SimpleLinearCRFModel
{
public: // Static functions
  /// BOSLabel() acts as the starting label
  static LabelID BOSLabel() { return BOS_LABEL; }

private: // Forward declarations
  template<unsigned O> friend class AveragedPerceptronCRFTrainer;
  template<unsigned O> friend class CRFTrainer;

private:
  typedef std::vector<LabelIDParameterIndexPairVector>          Transitions;
  typedef std::pair<LabelID,AttributeID>                        LabelIDAttributeIDPair;
  typedef std::pair<LabelID,LabelID>                            LabelIDPair;
  typedef std::vector<ParameterIndexVector>                     ParameterIndexMatrix;
  typedef std::pair<ParameterIndex,Weight>                      ParameterIndexWeightPair;
  typedef std::vector<ParameterIndexWeightPair>                 ParameterIndexWeightPairVector;
  typedef std::map<ParameterIndex,AttributeID>                  ParameterIndexAttributeIDMap;
  typedef std::map<ParameterIndex,LabelIDAttributeIDPair>       ParameterIndexToLabelIDAttributeIDPairMap;
  
  struct LabelIDPairHash {
    inline unsigned operator()(const LabelIDPair& y) const { return y.first * 18911 + y.second; }
  }; // LabelIDPairHash

  //typedef boost::unordered_map<LabelIDAttributeIDPair,ParameterIndex,LabelIDAttributeIDPairHash>  LabelAttributes;
  typedef boost::unordered_map<LabelIDPair,ParameterIndex,LabelIDPairHash>    TransitionWeights;
  typedef boost::unordered_map<AttributeID,ParameterIndex>                    AttributeIDParamIndexMap;

public:
  /// CRFState represents a state of a higher-order CRF. 
  /// It consists out of a static vector of size ORDER of label IDs.
  struct CRFHigherOrderState
  {
    /// Constructor
    CRFHigherOrderState(LabelID l = LabelID(-1)) 
    {
      // Create a state with history length 1, most often this is (BOS)
      std::fill(labels,labels+ORDER-1,LabelID(-1));
      labels[ORDER-1] = l;
      hist_len = 1;
    }

    inline const LabelID operator[](unsigned i) const { return labels[i]; }
    inline LabelID& operator[](unsigned i)            { return labels[i]; }
    inline LabelID label_id()                   const { return labels[ORDER-1]; }
    inline unsigned history_length()            const { return hist_len; }
    inline bool is_bos_state()                  const { return labels[ORDER-hist_len] == BOS_LABEL; }
    
    inline friend bool operator==(const CRFHigherOrderState& x,const CRFHigherOrderState& y)
    { return memcmp(&x,&y,ORDER*sizeof(LabelID)) == 0; }

    void construct(const LabelID* begin,const LabelID* end)
    {
      if (end-begin == ORDER) {
        hist_len = ORDER;
        memcpy(&labels[0],begin,hist_len*sizeof(LabelID));
      }
      else {
        hist_len = end-begin;
        std::fill_n(&labels[0],ORDER,LabelID(-1));
        memcpy(&labels[ORDER-hist_len],begin,hist_len*sizeof(LabelID));
        labels[ORDER-hist_len-1] = BOS_LABEL;
        ++hist_len;
      }
    }

    /// Shorten history
    void shorten_history()
    {
      if (hist_len == 0) return;
      labels[ORDER-hist_len] = LabelID(-1);
      --hist_len;
    }

    /// Moves all label IDs to the left
    inline CRFHigherOrderState wrap(LabelID r) const
    {
      CRFHigherOrderState n(*this);
      memmove(&n.labels[ORDER-hist_len],&n.labels[ORDER-hist_len+1],(hist_len-1)* sizeof(LabelID));
      n.labels[ORDER-1] = r;
      return n;
    }
    
    /// Append label r to a state which hasn't reached its full order
    inline CRFHigherOrderState increase_history(LabelID r) const
    {
      assert(hist_len<ORDER);
      CRFHigherOrderState n(*this);
      memmove(&n.labels[0],&n.labels[1],(ORDER-1)* sizeof(LabelID));
      n.labels[ORDER-1] = r;
      ++n.hist_len;
      return n;
    }

    /// Return the hash value of the state tuple
    unsigned hash_value() const
    {
      unsigned h = 0;
      for (unsigned i = 0; i < ORDER; ++i) 
        h ^= labels[i] + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }

    /// Output state tuple on 'out'
    std::ostream& print(std::ostream& o, const StringUnsignedMapper* labels_mapper=0) const
    {
      o << "(";
      for (unsigned i = 0; i < ORDER; ++i) {
        if (labels[i] != LabelID(-1)) {
          if (labels_mapper == 0) o << labels[i];
          else o << labels_mapper->get_string(labels[i]);
          if (i < ORDER-1) o << ",";
        }
      }
      return o << ")";
    }

    std::string as_string(const StringUnsignedMapper* labels_mapper) const
    {
      std::string s = "(";
      for (unsigned i = 0; i < ORDER; ++i) {
        if (labels[i] != LabelID(-1)) {
          s += labels_mapper->get_string(labels[i]);
          if (i < ORDER-1) s += ",";
        }
      }
      s += ")";
      return s;
    }

    LabelID         labels[ORDER];    ///< State tuple
    unsigned short  hist_len;         ///< Length of the actual history
  }; // CRFState

  struct CRFStateHash {
    inline unsigned operator()(const CRFHigherOrderState& q) const { return q.hash_value(); }
  }; // CRFStateHash

  /// CRFStateMapper: maps state tuples (for higher-order CRFs) to state IDs 
  /// (which are also label IDs) and vice versa
  class CRFStateMapper
  {
  public:
    /// Constructor
    CRFStateMapper(unsigned n = 0, const StringUnsignedMapper* lm = 0) : labels_mapper(lm) 
    {
      crf_states.reserve(n);
    }

    /// Maps a state tuple to a state ID
    inline CRFStateID operator()(const CRFHigherOrderState& q) const 
    {
      auto fq = state_to_id_map.find(q);
      return (fq != state_to_id_map.end()) ? fq->second : CRFStateID(-1);
    }

    /// Maps a state tuple q to a state ID. If q doesn't exist, it will be added and a new unique ID returned.
    inline CRFStateID operator()(const CRFHigherOrderState& q) 
    {
      auto fq = state_to_id_map.find(q);
      if (fq != state_to_id_map.end()) {
        return fq->second;
      }
      crf_states.push_back(q);
      state_to_id_map.insert(std::make_pair(q,crf_states.size()-1));
      return crf_states.size()-1;
    }

    /// ID --> state tuple
    inline const CRFHigherOrderState& operator()(CRFStateID qid) const
    {
      static CRFHigherOrderState invalid_state;
      return (qid < crf_states.size()) ? crf_states[qid] : invalid_state;
    }

    void print(std::ostream& o, const char* pref) const
    {
      for (unsigned i = 0; i < crf_states.size(); ++i) {
        o << pref << i << ": ";
        crf_states[i].print(o,labels_mapper);
        o << std::endl;
      }
    }

    /// Read a state mapper from a binary file stream
    bool read(std::ifstream& in)
    {
      unsigned order, n;
      in.read((char*)&order,sizeof(order));
      if (order != ORDER) {
        std::cerr << "";
        return false;
      }

      in.read((char*)&n,sizeof(n));
      crf_states.resize(n);
      in.read((char*)&crf_states[0],n*sizeof(CRFHigherOrderState));
      for (unsigned i = 0; i < crf_states.size(); ++i) {
        state_to_id_map.insert(std::make_pair(crf_states[i],i));
      }
      return true;
    }

    /// Write a state mapper to a binary file stream
    void write(std::ofstream& out) const
    {
      unsigned order = ORDER, n = crf_states.size();
      out.write((char*)&order,sizeof(order));
      out.write((char*)&n,sizeof(n));
      out.write((char*)&crf_states[0],n*sizeof(CRFHigherOrderState));
    }

    unsigned num_states() const { return state_to_id_map.size(); }

  private:
    const StringUnsignedMapper*                                         labels_mapper;
    std::vector<CRFHigherOrderState>                                    crf_states;
    boost::unordered_map<CRFHigherOrderState,CRFStateID,CRFStateHash>   state_to_id_map;
  }; // CRFStateMapper

  /// Iterator over ingoing/outgoing transitions
  class TransitionConstIterator
  {
  public:
    TransitionConstIterator() : params(0), transitions(0) {}
    TransitionConstIterator(const LabelIDParameterIndexPairVector& in_tr, const ParameterVector& p)
    : transitions(&in_tr), params(&p), current(in_tr.begin()) {}

    inline const TransitionConstIterator& operator++() { ++current; return *this; }

    LabelID from()  const { return current->first; }
    LabelID to()    const { return current->first; }
    Weight weight() const { return (*params)[current->second]; }

    inline const LabelIDWeightPair* operator->() const
    {
      current_label_weight.first = current->first;
      current_label_weight.second = (*params)[current->second];
      return &current_label_weight;
    }

    inline const LabelIDWeightPair& operator*() const
    {
      current_label_weight.first = current->first;
      current_label_weight.second = (*params)[current->second];
      return current_label_weight;
    }

    inline bool at_end() const { return current == transitions->end(); }

    friend bool operator==(const TransitionConstIterator& x,const TransitionConstIterator& y)
    { return x.current == y.current; }

    friend bool operator!=(const TransitionConstIterator& x,const TransitionConstIterator& y)
    { return !(x == y); }

  private:
    const ParameterVector*                          params;
    const LabelIDParameterIndexPairVector*          transitions;            ///< out/ingoing transitions
    LabelIDParameterIndexPairVector::const_iterator current;
    mutable LabelIDWeightPair                       current_label_weight;
  }; // TransitionConstIterator

public:
  /// Creates an empty model based on two mappings: a) labels and b) attributes
  SimpleLinearCRFModel(const StringUnsignedMapper& l_map, const StringUnsignedMapper& a_map)
  : labels_mapper(l_map), attributes_mapper(a_map), state_mapper(l_map.size(), &l_map), num_transitions(0),
    transitions(l_map.size()), labels_at_attributes(a_map.size()), good(true)
  {
    parameters.reserve(labels_mapper.size()*labels_mapper.size() + attributes_mapper.size() * 1.2);
    label_attributes.resize(labels_mapper.size());
  }

  /// Reads in a model from a text or binary stream
  SimpleLinearCRFModel(std::ifstream& in, bool binary=false) 
  : num_transitions(0), good(false)
  {
    good = binary ? read_model(in) : read_text_model(in);
    if (!good) 
      std::cerr << "Error: invalid model file\n";
  }
   
  /// Returns an iterator over the incoming transitions of y (this is used for first-order CRFs)
  inline TransitionConstIterator ingoing_transitions_of(LabelID y) const
  {
    static const LabelIDParameterIndexPairVector no_transitions;
    const LabelIDParameterIndexPairVector& in_tr = (y < transitions.size()) ? transitions[y] : no_transitions;
    return TransitionConstIterator(in_tr,parameters);
  }

  /// Returns an iterator over the outgoing transitions of y (this is used for higher-order CRFs)
  inline TransitionConstIterator outgoing_transitions_of(LabelID y) const
  {
    static const LabelIDParameterIndexPairVector no_transitions;
    const LabelIDParameterIndexPairVector& in_tr = (y < transitions.size()) ? transitions[y] : no_transitions;
    return TransitionConstIterator(in_tr,parameters);
  }

  /// Returns the parameter value at index p
  inline Weight operator[](ParameterIndex p) const
  {
    return (p < parameters.size()) ? parameters[p] : Weight(0.0);
  }

  inline Weight get_weight_for_attr_at_label(AttributeID a, LabelID y) const
  {
    AttributeIDParamIndexMap::const_iterator f = label_attributes[y].find(a);
    return (f != label_attributes[y].end()) ? parameters[f->second] : Weight(0.0); // TODO
  }

  /// Returns the parameter index for a feature
  inline ParameterIndex get_param_index_for_attr_at_label(AttributeID a, LabelID y) const
  {
    AttributeIDParamIndexMap::const_iterator f = label_attributes[y].find(a);
    return (f != label_attributes[y].end()) ? f->second : ParameterIndex(-1); // TODO
  }

  /// Returns the weight of a parameter with index p
  inline Weight weight_for_parameter(ParameterIndex p) const
  {
    return (p < parameters.size()) ? parameters[p] : Weight(0.0);
  }

  /// Returns the weight of the transition from y1 to y2
  inline Weight transition_weight(LabelID y1, LabelID y2) const
  {
    typename TransitionWeights::const_iterator ft = transition_weights.find(LabelIDPair(y1,y2));
    return ft != transition_weights.end() ? parameters[ft->second] : Weight(0.0);
  }

  /// Returns the parameter index of the transition from y1 to y2
  inline ParameterIndex transition_param_index(LabelID y1, LabelID y2) const
  {
    typename TransitionWeights::const_iterator ft = transition_weights.find(LabelIDPair(y1,y2));
    return ft != transition_weights.end() ? ft->second : ParameterIndex(-1); // TODO
  }

  /// Returns a vector of <label,param-index> pairs for those labels with which the attribute attr_id co-occurs
  inline const LabelIDParameterIndexPairVector& get_labels_for_attribute(AttributeID attr_id) const
  {
    static const LabelIDParameterIndexPairVector no_labels;
    return (attr_id < labels_at_attributes.size()) ? labels_at_attributes[attr_id] : no_labels;
  }

  /// Get the state tuple associated with a state ID (for hoCRFs)
  inline const CRFHigherOrderState& get_crf_state(CRFStateID q_id) const
  {
    static const CRFHigherOrderState invalid_state;
    return (ORDER > 1) ? state_mapper(q_id) : invalid_state;
  }

  /// Get the state ID associated with a state tuple (for hoCRFs)
  inline CRFStateID get_crf_state_id(const CRFHigherOrderState& q) const
  {
    return (ORDER > 1) ? state_mapper(q) : CRFStateID(-1);
  }

  /// Return some basic information about the model
  SimpleLinearCRFModelMetaData model_meta_data() const
  {
    SimpleLinearCRFModelMetaData md;
    md.order = ORDER;
    md.num_labels = labels_count();
    md.num_states = states_count();
    md.num_attributes = attributes_count();
    md.num_features = features_count();
    md.num_transitions = transitions_count();
    md.num_parameters = parameters_count();
    return md;
  }

  /// Return some basic information about the model in the binary file 'filename'
  SimpleLinearCRFModelMetaData model_meta_data(const std::string& filename) const
  {
    SimpleLinearCRFModelMetaData md;
    std::ifstream model_in(filename.c_str(),std::ios::binary);
    if (!model_in || !read_model_header(model_in,md)) return md;
    return md;
  }

  /// Write the model to a binary stream
  bool write_model(std::ofstream& out) const
  {
    // Write header
    SimpleLinearCRFModelMetaData meta_data;
    meta_data.order = ORDER;
    meta_data.num_labels = labels_mapper.size();
    meta_data.num_states = (ORDER == 1) ? meta_data.num_labels : state_mapper.num_states();
    meta_data.num_transitions = transitions_count();
    meta_data.num_attributes = attributes_mapper.size();
    meta_data.num_features = features_count();
    meta_data.num_parameters = parameters_count();
    meta_data.num_non_null_parameters = parameters_count(); // TODO

    out.write(MODEL_HEADER_ID,strlen(MODEL_HEADER_ID)+1);
    out.write((char*)&meta_data,sizeof(meta_data));
    
    // Create space of offsets
    long offset_labels=0, offset_transitions=0, offset_attrs=0, offset_label_attrs=0, offset_params=0;
    long offset_of_offsets = out.tellp();
    out.write((char*)&offset_labels,sizeof(offset_labels));
    out.write((char*)&offset_attrs,sizeof(offset_attrs));
    out.write((char*)&offset_transitions,sizeof(offset_transitions));
    out.write((char*)&offset_label_attrs,sizeof(offset_label_attrs));
    out.write((char*)&offset_params,sizeof(offset_params));

    // Write labels
    offset_labels = out.tellp();
    if (!labels_mapper.write(out)) {
      return false;
    }

    // Read state mapping for higher-order models
    if (ORDER > 1) {
      state_mapper.write(out);
    }

    // Write attributes
    offset_attrs = out.tellp();
    if (!attributes_mapper.write(out)) {
      return false;
    }

    // Write transitions
    offset_transitions = out.tellp();
    for (unsigned to = 0; to < states_count(); ++to) {
      size_t n = transitions[to].size();
      out.write((char*)&n,sizeof(n));
      if (n > 0) {
        out.write((char*)&transitions[to][0],n*sizeof(LabelIDParameterIndexPair));
      }
    }
    
    // Write label attributes
    offset_label_attrs = out.tellp();
    for (unsigned a_id = 0; a_id < labels_at_attributes.size(); ++a_id) {
      size_t n = labels_at_attributes[a_id].size();
      //std::cout << "n= " << n << std::endl;
      out.write((char*)&n,sizeof(n));
      if (n > 0) {
        const LabelIDParameterIndexPairVector& la = labels_at_attributes[a_id];
        out.write((char*)&la[0],n*sizeof(LabelIDParameterIndexPair));
      } // if (n > 0)
    } // for q

    // Compress parameters
    ParameterIndexWeightPairVector compressed_params;
    const ParameterVector& params = get_parameters();
    for (unsigned k = 0; k < params.size(); ++k) {
      if (params[k] != Weight(0.0)) {
        compressed_params.push_back(ParameterIndexWeightPair(k,params[k]));
      }
    }
    // Write parameters
    offset_params = out.tellp();
    unsigned compressed_params_size = compressed_params.size();
    out.write((char*)&compressed_params_size,sizeof(compressed_params_size));
    out.write((char*) &compressed_params[0], sizeof(ParameterIndexWeightPair) * compressed_params.size());

    // Rewind and write offsets
    out.seekp(offset_of_offsets);
    out.write((char*)&offset_labels,sizeof(offset_labels));
    out.write((char*)&offset_attrs,sizeof(offset_attrs));
    out.write((char*)&offset_transitions,sizeof(offset_transitions));
    out.write((char*)&offset_label_attrs,sizeof(offset_label_attrs));
    out.write((char*)&offset_params,sizeof(offset_params));

    out.close();
    return true;
  }

  // Read a model from a binary ifstream
  bool read_model(std::ifstream& in)
  {
    SimpleLinearCRFModelMetaData metadata;

    // Read header
    if (!read_model_header(in,metadata)) 
      return false;

    long offset_labels=0, offset_transitions=0, offset_attrs=0, offset_label_attrs=0, offset_params=0;
    in.read((char*)&offset_labels,sizeof(offset_labels));
    in.read((char*)&offset_attrs,sizeof(offset_attrs));
    in.read((char*)&offset_transitions,sizeof(offset_transitions));
    in.read((char*)&offset_label_attrs,sizeof(offset_label_attrs));
    in.read((char*)&offset_params,sizeof(offset_params));

    // Read labels
    if (!labels_mapper.read(in)) {
      std::cerr << "Error (SimpleLinearCRFModel::read_model()): "
                << "Unable to read the labels of the CRF model from the binary file\n";
      return false;
    }

    // Read state mapping for higher-order models
    if (ORDER > 1 && !state_mapper.read(in))  {
      return false;
    }

    // Read attributes
    if (!attributes_mapper.read(in)) {
      std::cerr << "Error (SimpleLinearCRFModel::read_model()): " 
                << "Unable to read the attributes of the CRF model from the binary file\n";
      return false;
    }

    // Read transitions
    num_transitions = metadata.num_transitions;
    transitions.resize(metadata.num_states);
    for (unsigned to = 0; to < metadata.num_states; ++to) {
      size_t n = 0;
      in.read((char*)&n,sizeof(n));
      if (n > 0) {
        transitions[to].resize(n);
        auto& trans_to = transitions[to];
        in.read((char*)&trans_to[0],n*sizeof(LabelIDParameterIndexPair));
        for (unsigned i = 0; i < n; ++i) {
          transition_weights.insert(std::make_pair(LabelIDPair(trans_to[i].first,to),trans_to[i].second));
        }
      } // if n > 0
    } // for to

    // Read labels
    labels_at_attributes.resize(metadata.num_attributes);
    label_attributes.resize(metadata.num_labels);
    for (unsigned a_id = 0; a_id < labels_at_attributes.size(); ++a_id) {
      size_t n = 0;
      in.read((char*)&n,sizeof(n));
      if (n > 0) {
        LabelIDParameterIndexPairVector& la = labels_at_attributes[a_id];
        la.resize(n);
        in.read((char*)&la[0],n*sizeof(LabelIDParameterIndexPair));
        for (unsigned i = 0; i < n; ++i) {
          label_attributes[la[i].first].insert(std::make_pair(a_id,la[i].second));
        }
      } // if (n > 0)
    } // for q

    // Read compressed params
    unsigned compressed_params_size = 0;
    in.read((char*)&compressed_params_size,sizeof(compressed_params_size));
    if (compressed_params_size > metadata.num_parameters) {
      std::cerr << "Error (SimpleLinearCRFModel::read_model()): Inconsistent model meta data\n";
      return false;
    }

    // Uncompress parameters
    ParameterIndexWeightPairVector compressed_params(compressed_params_size);
    in.read((char*) &compressed_params[0], sizeof(ParameterIndexWeightPair) * compressed_params.size());
    parameters.resize(metadata.num_parameters,Weight(0.0));
    for (unsigned k = 0; k < compressed_params_size; ++k) {
      parameters[compressed_params[k].first] = compressed_params[k].second;
    }

    return true;
  }

  /// Get the label ID for a label string
  inline LabelID get_label_id(const Label& label) const
  {
    return labels_mapper.get_id(label);
  }

  /// Get the attribute ID for an attribute string
  inline AttributeID get_attr_id(const Attribute& attr) const
  {
    return attributes_mapper.get_id(attr);
  }

  /// Get the label string for a label ID
  const Label& get_label(LabelID id) const
  {
    return labels_mapper.get_string(id);
  }

  /// Get the attribute string for an attribute ID
  const Attribute& get_attr(AttributeID id) const
  {
    return attributes_mapper.get_string(id);
  }

  /// Get the label ID for <BOS>
  inline LabelID get_bos_label_id() const
  {
    return 0;
  }

  /// Return the number of different feature functions
  /// Note that a feature is a distinct attribute-label combination
  unsigned features_count() const 
  { 
    unsigned num_features = 0;
    for (unsigned l = 0; l < label_attributes.size(); ++l) {
      num_features += label_attributes[l].size();
    }
    return num_features; 
  }

  unsigned labels_count()       const { return label_attributes.size(); }
  unsigned states_count()       const { return (ORDER == 1) ? labels_count() : state_mapper.num_states(); }
  unsigned attributes_count()   const { return attributes_mapper.size(); }
  unsigned transitions_count()  const { return num_transitions; }
  unsigned parameters_count()   const { return parameters.size(); }
  unsigned model_order()        const { return ORDER; }
  
  /// Returns the start state of the model. This is currently only meaningful for higher-order CRFs 
  /// in which case it is state <BOS> with ID 0 (this must ensured by the training algorithm)
  CRFStateID start_state()      const { return (ORDER > 1) ? 0 : CRFStateID(-1); }
  
  /// Saves memory by returning unallocated vector space
  void finalise(bool compress_params=true)
  {
    if (compress_params) 
      ParameterVector(parameters).swap(parameters);
    //std::cerr << "parameters.size() == " << parameters.size() << ", parameters.capacity() == " << parameters.capacity() << "\n";
    for (unsigned a = 0; a < labels_at_attributes.size(); ++a) {
      if (labels_at_attributes[a].size() != labels_at_attributes[a].capacity()) {
        LabelIDParameterIndexPairVector(labels_at_attributes[a]).swap(labels_at_attributes[a]);
      }
    }
  }

  /// Read-only access to the parameters
  const ParameterVector& get_parameters() const { return parameters; }

  /// Outputs model in textual form on 'out'
  friend std::ostream& operator<<(std::ostream& out, const SimpleLinearCRFModel& m)
  {
    return m.print(out);
  }

  /// Outputs a graphviz dot representation of the transitions to 'out'
  void draw(std::ostream& out) const
  {
    static std::string NodeColors[] = {"","cornflowerblue","blue","navyblue","slateblue","turquoise","indigo","green"};

 	  out << "digraph G {\n";
	  out << "graph [rankdir=LR, fontsize=14, center=1, orientation=Portrait];\n";
	  out << "node  [font = \"Arial\", shape = circle, style=filled, fontcolor=white, color=blue]\n";
	  out << "edge  [fontname = \"Arial\"]\n\n";

    if (ORDER == 1) {
      // First-order transitions
      for (unsigned to = 1; to < labels_mapper.size(); ++to) {
        out << "\t" << to << " [label=\"" << get_label(to) << "\"]" << std::endl; 
        for (TransitionConstIterator t = ingoing_transitions_of(to); !t.at_end(); ++t) {
          out << "\t" << t.from() << " -> " << to << " [label=\"" << t.weight() << "\"]\n";
        } // for t
      } // for to
    }
    else {
      // Higher-order transitions
      // 1. Group states by history length
      std::map<unsigned,std::vector<CRFHigherOrderState> > subgraphs;
      for (CRFStateID q_id = 0; q_id < state_mapper.num_states(); ++q_id) {
        CRFHigherOrderState q = state_mapper(q_id);
        subgraphs[q.history_length()].push_back(q);
      }

      // 2. Process state groups
      for (auto sg = subgraphs.rbegin(); sg != subgraphs.rend(); ++sg) {
        std::string ncolor = (sg->first < sizeof(NodeColors)/sizeof(NodeColors[0])) ? NodeColors[sg->first] : "slategrey";
        out << "subgraph cluster" << (sg->first) << " {" << std::endl;
	      out << "  node [color=\"" << ncolor << "\"]" << std::endl;
        for (unsigned i = 0; i < sg->second.size(); ++i) {
          const CRFHigherOrderState& from = sg->second[i];
          CRFStateID from_id = state_mapper(from);
          out << "  " << from_id << " [label=\"";
          from.print(out,&labels_mapper);
          out << "\"]" << std::endl;
          // Iterate over all transitions entering to_id
          for (TransitionConstIterator t = outgoing_transitions_of(from_id); !t.at_end(); ++t) {
            CRFHigherOrderState to = state_mapper(t.to());
            CRFStateID to_id = state_mapper(to);
            std::string trans_color;
            if (to.history_length() == from.history_length())     trans_color = "black";
            else if (to.history_length() > from.history_length()) trans_color = "blue";
            else if (to.history_length() < from.history_length()) trans_color = "green";
            out << "\t" << from_id << " -> " << to_id 
                << " [label=\"" << get_label(to.label_id()) 
                << " / " << t.weight() << "\","
                << "style=" << "bold" << ","
                << "color=" << trans_color
                << "]" << std::endl;            
          } // for t
        } // for i
        out << "}\n\n";
      } // for sg
    }

	  out << "}\n";
  }

private: // Functions
  bool read_model_header(std::ifstream& in, SimpleLinearCRFModelMetaData& meta_data)
  {
    // Read header
    char model_id[100];

    in.read(model_id,strlen(MODEL_HEADER_ID)+1);
    if (std::string(model_id) != std::string(MODEL_HEADER_ID)) {
      std::cerr << "Error (SimpleLinearCRFModel::read_model()): Invalid binary model file\n";
      return false;
    }

    in.read((char*)&meta_data,sizeof(meta_data));

    if (meta_data.order != ORDER) {
      std::cerr << "Error (SimpleLinearCRFModel::read_model()): Incompatible model orders\n";
      return false;
    }

    // Some plausability tests
    if ((meta_data.num_parameters != meta_data.num_transitions + meta_data.num_features) ||
        (meta_data.num_attributes >=  meta_data.num_features) ||
        (meta_data.num_transitions > meta_data.num_states*meta_data.num_states)) {
      std::cerr << "Error (SimpleLinearCRFModel::read_model()): Inconsistent model meta data\n";
      return false;
    }

    return true;
  }

  /// Reads a model in CRFSuite dump format from a text file
  bool read_text_model(std::istream& in)
  {
    typedef boost::char_separator<char>     CharSeparator;
    typedef boost::tokenizer<CharSeparator> Tokenizer;
    
    const CharSeparator segment_at_colon("\t ",":");
    const std::string Colon(":"); 
    const std::string Arrow("-->");

    enum { qIntermediate, qHeader, qLabels, qFeatures, qAttributes, qTransitions, qStateFeatures, qStop } current_state;
    std::string line;
    current_state = qIntermediate;
    std::vector<std::string> tokens;
    unsigned read_features = 0, read_attributes = 0, read_labels = 0, num_labels2 = 0, num_attrs2 = 0;
    unsigned line_no = 0;

    while (in.good()) {
      std::getline(in,line);
      ++line_no;
      if (line.empty()) continue;
      
      if (current_state == qIntermediate) {
        if      (line == "FILEHEADER = {")      current_state = qHeader;
        else if (line == "LABELS = {")          current_state = qLabels;
        else if (line == "ATTRIBUTES = {")      current_state = qAttributes;
        else if (line == "TRANSITIONS = {")     current_state = qTransitions;
        else if (line == "STATE_FEATURES = {")  current_state = qStateFeatures;
      }

      else if (current_state == qHeader) {
        if (line == "}") current_state = qIntermediate;
        else {
          Tokenizer tokenizer(line,segment_at_colon);
          tokens.assign(tokenizer.begin(),tokenizer.end());
          if (tokens.size() == 3 && tokens[1] == Colon) {
            if (tokens[0] == "model_order") {
              unsigned o = boost::lexical_cast<unsigned>(tokens[2]);
              if (o != ORDER) {
                std::cerr << "SimpleLinearCRFModel::read_text_model(): Incompatible model orders." << std::endl;
                return false;
              }
            }
            else if (tokens[0] == "num_features") {
              //num_features = boost::lexical_cast<unsigned>(tokens[2]);
            }
            else if (tokens[0] == "num_labels") {
              num_labels2 = boost::lexical_cast<unsigned>(tokens[2]);
              set_labels(num_labels2);
            }
            else if (tokens[0] == "num_states") {
              // $TODO
            }
            else if (tokens[0] == "num_attrs") {
              num_attrs2 = boost::lexical_cast<unsigned>(tokens[2]);
              set_attributes(num_attrs2);
            }
            else if (tokens[0] == "num_trans") {
              // Not a standard attribute
            }
          }  
        }
      } // current_state == qHeader

      else if (current_state == qLabels) {
        if (line == "}") current_state = qIntermediate;
        else {
          Tokenizer tokenizer(line,segment_at_colon);
          tokens.assign(tokenizer.begin(),tokenizer.end());
          if (tokens.size() == 3 && tokens[1] == Colon) {
            // Label = 2, ID = 0
            add_label(tokens[2],boost::lexical_cast<unsigned>(tokens[0]));
            ++read_labels;
          }
          else std::cerr << "SimpleLinearCRFModel::read_text_model(): Error in line " << line_no << ": invalid label entry." << "\n";
        }
      } // current_state == qLabels

      else if (current_state == qAttributes) {
        if (line == "}") current_state = qIntermediate;
        else {
          Tokenizer tokenizer(line,segment_at_colon);
          tokens.assign(tokenizer.begin(),tokenizer.end());
          if (tokens.size() == 3 && tokens[1] == Colon) {
            // Attribute = 2, ID = 0
            add_attr(tokens[2],boost::lexical_cast<unsigned>(tokens[0]));
            ++read_attributes;
          }
          else std::cerr << "Error in line '" << line << "': invalid attribute entry." << "\n";
        }
      } // current_state == qAttributes

      else if (current_state == qTransitions) {
        if (line == "}") {
          current_state = qIntermediate;
        }
        else {
          Tokenizer tokenizer(line,segment_at_colon);
          tokens.assign(tokenizer.begin(),tokenizer.end());
          //   (1) None --> None: -0.044628
          if (tokens.size() == 6 && tokens[4] == Colon && tokens[2] == Arrow) {
            // ? = 0 , Label1 = 1, Arrow = 2, Label2 = 3, Colon=4, Param = 5
            int y1 = get_label_id(tokens[1]);
            int y2 = get_label_id(tokens[3]);
            if (y1 != -1 && y2 != -1) {
              add_transition(y1,y2,boost::lexical_cast<float>(tokens[5]));
            }
            else std::cerr << "Error in line '" << line << "': invalid transition entry." << "\n";
          }
        }
      } // current_state == qTransitions

      else if (current_state == qStateFeatures) {
        if (line == "}") current_state = qStop;
        else {
          Tokenizer tokenizer(line,segment_at_colon);
          tokens.assign(tokenizer.begin(),tokenizer.end());
          //  (0) type[-1]|type[0]=InitUpper|InitUpper --> NELN: 0.039935
          if (tokens.size() == 6 && tokens[4] == Colon && tokens[2] == Arrow) {
            // ? = 0 , Attr = 1, Arrow = 2, Label = 3, Colon=4, Param = 5
            int attr = get_attr_id(tokens[1]);
            int y = get_label_id(tokens[3]);
            if (attr != -1 && y != -1) {
              add_attr_for_label(y,attr,boost::lexical_cast<float>(tokens[5]));
            }
            else std::cerr << "Error in line '" << line << "': invalid label features entry." << "\n";
          }
        }
      } // current_state == qStateFeatures
    } // while

    finalise();

    // Sort transitions
    for (auto t = transitions.begin(); t != transitions.end(); ++t) {
      std::sort(t->begin(),t->end());
    }
    
    return true;
    //return (current_state == qStop) && (read_features == num_features) &&
    //       (read_labels == num_labels) && (read_attributes == num_attrs);
  }
  
  std::ostream& print(std::ostream& out) const
  {
    typedef std::map<unsigned,std::string> IDStringMap;

    out.precision(8);

    out << "FILEHEADER = {" << std::endl;
    out << "  model_type: " << "crf_hmm" << std::endl;
    out << "  model_order: " << ORDER << std::endl;
    out << "  num_features: " << features_count() << std::endl;
    out << "  num_labels: " << labels_mapper.size() << std::endl;
    if (ORDER > 1) {
      out << "  num_states: " << state_mapper.num_states() << std::endl;
    }
    out << "  num_attrs: " << attributes_mapper.size() << std::endl;
    out << "  num_transitions: " << num_transitions << std::endl;
    out << "  num_params: " << parameters.size() << std::endl;
    out << "}" << std::endl << std::endl;
   
    out << "LABELS = {" << std::endl;
    this->labels_mapper.print(out,"  ", ": ");
    out << "}" << std::endl << std::endl;

    if (ORDER > 1) {
      out << "STATES = {" << std::endl;
      state_mapper.print(out,"  ");
      out << "}" << std::endl << std::endl;
    }

    out << "ATTRIBUTES = {" << std::endl;
    this->attributes_mapper.print(out,"  ", ": ");
    out << "}" << std::endl << std::endl;
 
    out << "TRANSITIONS = {" << std::endl;
    print_transitions(out);
    out << "}" << std::endl << std::endl;

    out << "STATE_FEATURES = {" << std::endl;
    for (unsigned q = 0; q != label_attributes.size(); ++q) {
      for (auto la = label_attributes[q].begin(); la != label_attributes[q].end(); ++la) {
        if (parameters[la->second] != 0.0) {
          out << "  " << "(0) " 
              << get_attr(la->first) << " --> " << get_label(q) << ": "
              << std::setprecision(7) << parameters[la->second] << std::endl;
        } // if 
      } // for la
    } // for q
    return out << "}" << std::endl << std::endl;
  }

  void print_transitions(std::ostream& out) const
  {
    if (ORDER == 1) {
      // First-order transitions
      for (unsigned to = 0; to < labels_mapper.size(); ++to) {
        const LabelIDParameterIndexPairVector& trans = transitions[to];
        for (unsigned j = 0; j < trans.size(); ++j) {
          // (1) OTHER --> ORG_B: 0.482204
          out << "  (1) " 
              << get_label(trans[j].first) << " --> "
              << get_label(to)
              << ": " << parameters[trans[j].second] 
              << std::endl;
        }
      } // for to
    }
    else {
      // Higher-order transitions
      for (CRFStateID from_id = 0; from_id < state_mapper.num_states(); ++from_id) {
        CRFHigherOrderState from = state_mapper(from_id);
        for (TransitionConstIterator t = outgoing_transitions_of(from_id); !t.at_end(); ++t) {
          CRFHigherOrderState to = state_mapper(t.to());
          out << "  (1) ";
          from.print(out,&labels_mapper);
          out << " --> ";
          to.print(out,&labels_mapper);
          out << ": " << t.weight() << std::endl;
        } // for t
      } // for from_id
    }
  }   
 
  bool add_label(const Label& label, unsigned id) 
  {
    return labels_mapper.add_pair(label,id);
  }
    
  bool add_attr(const Attribute& attr, unsigned id)
  {
    return attributes_mapper.add_pair(attr,id);
  }

  /// Adds a transition "from --> to" with weight
  bool add_transition(LabelID from, LabelID to, Weight weight=Weight(0.0)) 
  {
    typename TransitionWeights::const_iterator f = transition_weights.find(LabelIDPair(from,to));
    if (f == transition_weights.end()) {
      // New transition
      if (to >= transitions.size()) {
        transitions.resize(to*2+1);
      }
      transitions[to].push_back(LabelIDParameterIndexPair(from,parameters.size()));
      transition_weights.insert(std::make_pair(LabelIDPair(from,to),parameters.size()));
      //if (ORDER > 1)
      //  std::cerr << "\nAdding transition " << to << " --> " << from << " with index " << parameters.size();
      parameters.push_back(weight);
      ++num_transitions;
      return true;
    }
    return false;
  }

  /// Adds a transition between state tuples (for hoCRF)
  /// @note Do not mix this with add_transition(LabelID,LabelID,Weight)
  bool add_transition(const CRFHigherOrderState& from, const CRFHigherOrderState& to, Weight weight=Weight(0.0))
  {
    auto from_id = state_mapper(from); // This ensures that <BOS> will be the first mapped state 
    auto to_id = state_mapper(to);
    // Note: higher-order CRFs are based on outgoing transitions
    if (ORDER == 1) return add_transition(from_id,to_id,weight);
    else            return add_transition(to_id,from_id,weight); 
  }

  /// Associate an attribute with a label (in other words: create a feature)
  void add_attr_for_label(LabelID label_id, AttributeID attr_id, Weight weight=Weight(0.0))
  {
    static struct LabelIDParameterIndexPairLess {
      inline bool operator()(const LabelIDParameterIndexPair& x, const LabelIDParameterIndexPair& y) const
      { return x.first < y.first; }
    } cmp; // LabelIDParameterIndexPairLess
    
    //static LabelIDParameterIndexPairLess cmp;

    // Store attr_id as an observed attribute at label with label_id
    label_attributes[label_id].insert(std::make_pair(attr_id,parameters.size()));
    
    // Store label_id as an observed label with attr_id
    LabelIDParameterIndexPairVector& la = labels_at_attributes[attr_id];

    // Binary search
    auto pos = std::lower_bound(la.begin(),la.end(),LabelIDParameterIndexPair(label_id,parameters.size()),cmp);
    if (pos == la.end() || pos->first != label_id) {
      // Not yet present
      la.insert(pos,LabelIDParameterIndexPair(label_id,parameters.size()));
      parameters.push_back(weight);
    }
  }
  
  /// For the purpose of training: Training algorithms have friend access to the parameters
  ParameterVector& get_parameters() { return parameters; }

  /// For debugging purposes
  const StringUnsignedMapper& get_labels_mapper() const { return labels_mapper; }

  /// Replace the parameter vector by one externally computed of the same size
  void set_parameters(const ParameterVector& new_params) 
  { 
    if (new_params.size() == parameters.size()) {
      // TODO: replace that by swap()
      parameters.assign(new_params.begin(),new_params.end());
    }
    else {
      std::cerr << "SimpleLinearCRFModel: New parameters vector in set_parameters() has different size\n";
    }
  }

  /// Set the number of labels of the model
  void set_labels(unsigned n)
  {
    transitions.resize(n);
    label_attributes.resize(n);
  }

  /// Set the number of attributes of the model
  void set_attributes(unsigned n)
  {
    labels_at_attributes.resize(n);
    parameters.reserve(n);
  }

private:
  // String-ID-infrastructure
  StringUnsignedMapper                          labels_mapper;        ///< Map labels <-> label IDs
  StringUnsignedMapper                          attributes_mapper;    ///< Map attributes <-> attribute IDs
  
  // Mapping of complex states (for higher-order CRFs)
  CRFStateMapper                                state_mapper;         ///< Map state tuples <-> state tuple IDs

  // Parameter-related variables
  Transitions                                   transitions;          ///< Transition matrix
  TransitionWeights                             transition_weights;   ///< Adjacency matrix
  ParameterVector                               parameters;           ///< All model parameters reside here
  std::vector<AttributeIDParamIndexMap>         label_attributes;     ///<
  std::vector<LabelIDParameterIndexPairVector>  labels_at_attributes; ///<
  ParameterIndexToLabelIDAttributeIDPairMap     param_to_attr;        ///<

  // Model meta data
  unsigned                                      num_transitions;      ///< Number of transitions
  bool                                          good;
}; // SimpleLinearCRFModel

#endif
