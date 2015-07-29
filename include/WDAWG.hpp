////////////////////////////////////////////////////////////////////////////////
// WDAWG.hpp
// Implements weighted directed acyclic word graphs
// TH, Sept. 2014
////////////////////////////////////////////////////////////////////////////////

#ifndef __WDAWG_HPP__
#define __WDAWG_HPP__

#include <map>
#include <boost/container/flat_map.hpp>
#include <set>
#include <vector>
#include <iostream>
#include <stack>
//#define NDEBUG
#include <cassert>

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>

#define BINARY_WDAWG_HEADER    "Binary wdawg file"


/**
  \brief IncrementalMinimizer implements the incremental minimisation algorithm after Daciuk et al. (2000)
         (algorithm 1 from sorted data)
  \pre an entry must contain the output data at the first position
  \todo Compact delta function
*/
template< typename SYMBOL, 
          typename FINAL_INFO, 
          typename LABEL_SERIALISER, 
          typename FINAL_INFO_SERIALISER=LABEL_SERIALISER >
class WeightedDirectedAcyclicWordGraph
{
  public: // Types
    typedef int                                             State;
    typedef SYMBOL                                          Symbol;
    typedef FINAL_INFO                                      FinalInfo;
    typedef std::vector<Symbol>                             SymbolVector;
    typedef std::pair<SymbolVector,FinalInfo>               Entry;
    typedef std::vector<Entry>                              EntryVector;
    typedef std::set<FinalInfo>                             FinalStateInfoSet;

  public: // Static functions
    inline static State NoState() { return -1; }

  private: // Types
    typedef boost::unordered_set<State>                     StateSet;
    typedef std::pair<State,unsigned>                       StateIndexPair; // (State, entry position)
    typedef int                                             SymbolIndex;
    typedef boost::container::flat_map<Symbol,State>        SymbolStateMap;
    typedef std::vector<SymbolStateMap>                     Delta;
    typedef std::stack<State>                               StateStack;
    typedef boost::unordered_map<State,FinalStateInfoSet>   FinalInfoMap;

  public:
    WeightedDirectedAcyclicWordGraph()
    : state_register(180811,StateHash(delta,final_states),StateEquiv(delta,final_states))
    {
      // Create start state
      new_state();
    }

    /// Construct a weighted DAWG from a sorted list
    /// Precondition: entries in 'entries' must be sorted (after the same criterion as SymbolStateMap)
    WeightedDirectedAcyclicWordGraph(const EntryVector& entries)
    : state_register(180811,StateHash(delta,final_states),StateEquiv(delta,final_states))
    {
      // Create start state
      new_state();
      process(entries);
    }

    /// Constructs an empty DAWG from a binary file stream
    WeightedDirectedAcyclicWordGraph(std::ifstream& in)
    : state_register(180811,StateHash(delta,final_states),StateEquiv(delta,final_states))
    {
      bool good = read(in);
    }

    /// Returns the start state
    inline State start_state() const
    {
      return 0;
    }

    /// Returns the number of the final states in the FSA
    inline unsigned no_of_final_states() const
    {
      return final_states.size();
    }

    /// Returns the number of the states in the FSA
    inline unsigned no_of_states() const
    {
      return delta.size();
    }

    /// Returns the number of the transitions in the FSA
    inline unsigned no_of_transitions() const
    {
      unsigned nt = 0;
      for (auto s = state_register.begin(); s != state_register.end(); ++s) {
        nt += delta[*s].size();
      }
      return nt;
    }

    inline const FinalStateInfoSet& final_info(State q) const
    {
      static const FinalStateInfoSet no_info;
      auto fq = final_states.find(q);
      return fq != final_states.end() ? fq->second : no_info;
    }

    /// Returns the out-degree of the FSA
    inline unsigned out_degree() const
    {
      unsigned od = 0;
      for (auto s = state_register.begin(); s != state_register.end(); ++s) {
        if (delta[*s].size() > od)
          od = delta[*s].size();
      }
      return od;
    }

    /// Print a dot representation of the FSA to stream 'out'
    void draw(std::ofstream& out) const
    {
      typedef typename SymbolStateMap::const_iterator TransitionsIter;

      out << "digraph FSM {\n";
      out << "graph [rankdir=LR, fontsize=14, center=1, orientation=Portrait];\n";
      out << "node  [font = \"Arial\", shape = circle, style=filled, fontcolor=black, color=lightgray]\n";
      out << "edge  [fontname = \"Arial\"]\n\n";

      for (unsigned q = 0; q < delta.size(); ++q) {
        out << q << " [label = \"" << q << "\"";
        out << (is_final(State(q)) ? ", shape=doublecircle]\n" : "]\n");
        const SymbolStateMap& q_tr = delta[q];
        for (TransitionsIter t = q_tr.begin(); t != q_tr.end(); ++t) {
          out << q << " -> " << t->second << " [label = \"" << t->first << "\"]\n";
        }
      }
      out << "}\n";
    }

    /// Write dawg to a binary file (todo: write type info)
    bool write(std::ostream& out) const
    {
      LABEL_SERIALISER label_serialiser;
      FINAL_INFO_SERIALISER final_info_serialiser;
      
      out.write(BINARY_WDAWG_HEADER, std::string(BINARY_WDAWG_HEADER).size()+1);
      unsigned nstates = delta.size();
      out.write((char*) &nstates, sizeof(nstates));
      nstates = final_states.size();
      out.write((char*) &nstates, sizeof(nstates));
      for (unsigned q = 0; q < delta.size(); ++q) {
        unsigned num_trans = delta[q].size();
        out.write((char*) &num_trans, sizeof(num_trans));
        for (auto t = delta[q].begin(); t != delta[q].end(); ++t) {
          // Write transition symbol
          label_serialiser.write(out,t->first);
          // write target state
          out.write((char*) &t->second, sizeof(t->second));
        }
      } // for q
    
      for (auto f = final_states.begin(); f != final_states.end(); ++f) {
        out.write((char*) &f->first, sizeof(f->first));
        unsigned short n = f->second.size();
        out.write((char*) &n, sizeof(n));
        for (auto i = f->second.begin(); i != f->second.end(); ++i) {
          final_info_serialiser.write(out,*i);
        }
      }
      return true;
    }

    /// Read dawg from a binary file
    bool read(std::istream& in)
    {
      LABEL_SERIALISER label_serialiser;
      FINAL_INFO_SERIALISER final_info_serialiser;

      char header[100];
      in.read(header, std::string(BINARY_WDAWG_HEADER).size()+1);
      if (std::string(header) != std::string(BINARY_WDAWG_HEADER)) {
        std::cerr << "Invalid input stream\n";
        return false;
      }

      clear();
      unsigned n_states, n_final_states;
      in.read((char*) &n_states, sizeof(n_states));
      in.read((char*) &n_final_states, sizeof(n_final_states));
      delta.resize(n_states);
      for (unsigned q = 0; q < delta.size(); ++q) {
        unsigned num_trans;
        in.read((char*) &num_trans, sizeof(num_trans));
        for (unsigned i = 0; i < num_trans; ++i) {
          // Read transition symbol
          Symbol l;
          label_serialiser.read(in,l);
          // Read target state
          State next;
          in.read((char*) &next, sizeof(next));
          delta[q].insert(std::make_pair(l,next));
        }
      } // for q

      for (unsigned i = 0; i < n_final_states; ++i) {
        typename FinalInfoMap::value_type state_info_pair;
        in.read((char*) &state_info_pair.first, sizeof(state_info_pair.first));
        unsigned short n;
        in.read((char*) &n, sizeof(n));
        for (unsigned k = 0; k < n; ++k) {
          FINAL_INFO fi;
          final_info_serialiser.read(in,fi);
          state_info_pair.second.insert(fi);
        }
        final_states.insert(state_info_pair);
      }
      return true;
    }

    /// Find target state p of the transition q --a-> p .
    /// Returns NoState() if p is undefined.
    inline State find_transition(State q, Symbol a) const
    {
      assert(q < delta.size());
      const SymbolStateMap& delta_q = delta[q];
      typename SymbolStateMap::const_iterator f = delta_q.find(a);
      return (f != delta_q.end()) ? f->second : NoState();
    }

    /// Returns true iff q is final
    inline bool is_final(State q) const
    {
      return final_states.find(q) != final_states.end();
    }

    void clear()
    {
      delta.clear();
      final_states.clear();
      state_register.clear();
      //free_list.clear();
    }

  private: // Functions
    /// Process the sorted entries in the vector
    void process(const EntryVector& entries)
    {
      // See algorithm 1 in Daciuk et al.
      delta.reserve(entries.size());
      unsigned n = 0;
      for (auto e = entries.begin(); e != entries.end(); ++e) {
        StateIndexPair si = common_prefix(e->first);
        if (has_children(si.first)) {
          replace_or_register(si.first);
        }
        add_suffix(si,*e);
      }
      replace_or_register(0);
      std::cerr << "# distinct_symbols = " << distinct_symbols.size() << std::endl;
    }

    /// Find the common prefix of 'entry' with a word already in the automaton
    StateIndexPair common_prefix(const SymbolVector& seq) const
    {
      /// Note the entry 0 always contains the data associated with the entry
      State current = 0;
      for (unsigned i = 0; i < seq.size(); ++i) {
        State p = find_transition(current,seq[i]);
        if (p == NoState()) {
          return StateIndexPair(current,i);
        }
        current = p;
      }
      return StateIndexPair(current,seq.size());
    }

    /// Returns true iff state q has outgoing transitions
    inline bool has_children(State q) const
    {
      assert(q >= 0 && q < delta.size()); 
      return delta[q].size() > 0;
    }

    /// See Daciuk et al. (2000)
    void replace_or_register(State p)
    {
      // Recurse to lexicographic last child (postorder traversal)
      State child = last_child(p);
      if (has_children(child)) {
        replace_or_register(child);
      }
      // Try to find a state in the register with same right language
      State q = equivalent_state_in_register(child);
      if (q != NoState()) {
        // There is an equivalent state
        replace_state(p,q);
        delete_state(child);
      }
      else {
        // There is no state with the same right language => insert child into the register
        state_register.insert(child);
      }
    }

    /// Adds the suffix entry[q_i.second ... entry.size()] to the FSA, starting at q_i.first
    void add_suffix(const StateIndexPair& q_i, const Entry& entry)
    {
      State q = q_i.first;
      for (unsigned i = q_i.second; i < entry.first.size(); ++i) {
        q = add_transition(q,entry.first[i]);
      }
      make_final(q,entry.second);
    }

    /// Looks up a state with the same right language as q
    inline State equivalent_state_in_register(State q) const
    {
      typename StateRegister::const_iterator fq = state_register.find(q);
      return fq != state_register.end() ? *fq : NoState();
    }

    /// Makes q final
    inline void make_final(State q, const FINAL_INFO& info)
    {
      final_states[q].insert(info);
    }

    /// Add a transition from q with symbol a to an unused state
    inline State add_transition(State q, Symbol a)
    {
      assert(q >= 0 && q < delta.size());
      State r = new_state();
      delta[q].insert(std::make_pair(a,r));
      distinct_symbols.insert(a);
      return r;
    }

    /// Find the lexicographic last child of q
    inline State last_child(State q) const
    {
      assert(q >= 0 && q < delta.size());
      // Since delta[q] is a map, rbegin() does the job
      return (has_children(q)) ? delta[q].rbegin()->second : NoState();
    }

    /// Returns an unused state
    inline State new_state()
    {
      if (free_list.empty()) {
        delta.push_back(SymbolStateMap());
        return delta.size()-1;
      }
      else {
        // Take a state from the free list
        State n = free_list.top();
        free_list.pop();
        return n;
      }
    }

    inline void replace_state(State p, State q)
    {
      typename SymbolStateMap::reverse_iterator f = delta[p].rbegin();
      if (f != delta[p].rend()) {
        f->second = q;
      }
    }

    /// Delete state by clearing its transition map and putting it on the free list
    inline void delete_state(State q)
    {
      if (q < delta.size()) {
        delta[q].clear();
        final_states.erase(q);
        // Put state on free list
        free_list.push(q);
      }
    }

  private:
    /// Hash object for states
    struct StateHash
    {
      StateHash(const Delta& d, const FinalInfoMap& f) 
      : delta(d), final_states(f) {}

      inline unsigned operator()(State q) const
      {
        unsigned hash = 0x9e3779b9;
        typename FinalInfoMap::const_iterator fq = final_states.find(q);
        if (fq != final_states.end()) {
          hash += boost::hash_range(fq->second.begin(),fq->second.end());
        }
        hash = (hash << 6) ^ (hash << 23) - hash;
        const SymbolStateMap& q_trans = delta[q];
        for (typename SymbolStateMap::const_iterator it = q_trans.begin(); it != q_trans.end(); ++it) {
          hash = (symbol_hasher(it->first) + (hash << 6) ^ (hash >> 16)) - hash;
          hash += (it->second + (hash << 6) ^ (hash << 16)) - hash;
        }
        return hash;
      }

      const Delta& delta;
      const FinalInfoMap& final_states;
      std::hash<Symbol> symbol_hasher;
    }; // StateHash

    /// Compare states by their right languages
    struct StateEquiv
    {
      /// Constructor
      StateEquiv(const Delta& d, const FinalInfoMap& fs) : delta(d), final_states(fs) {}

      /// Equivalence of states based on the 4 criteria in Daciuk et al.
      inline bool operator()(State p, State q) const
      {
        assert(p >= 0 && p < delta.size());
        assert(q >= 0 && q < delta.size());

        // Check for differing finality
        typename FinalInfoMap::const_iterator fp = final_states.find(p);
        typename FinalInfoMap::const_iterator fq = final_states.find(q);

        if (fq == final_states.end() && fp != final_states.end())
          return false;
        if (fp == final_states.end() && fq != final_states.end())
          return false;

        // Compare final state infos
        if (fp != final_states.end() && fq != final_states.end() && fp->second != fq->second)
          return false;

        // pairwise comparision => already built-in in map<>
        return delta[p] == delta[q];
      }

      const Delta& delta;
      const FinalInfoMap& final_states;
    }; // StateEquiv

    // StateRegister is defined as an unordered_set based on StateHash and StateEquiv
    typedef boost::unordered_set<State,StateHash,StateEquiv> StateRegister;

  private: // Member variables
    Delta                         delta;              ///< Delta-function
    FinalInfoMap                  final_states;       ///< Mapping from final states to associated information
    StateRegister                 state_register;     ///< State register based on right-languages-equality
    StateStack                    free_list;          ///< List of free (=usable) states
    boost::unordered_set<Symbol>  distinct_symbols;
}; // WeightedDirectedAcyclicWordGraph

template<typename LENTYPE>
struct StringSerialiser
{
  unsigned read(std::istream& i, std::string& s) const
  {
    LENTYPE len = 0;
    i.read((char*) &len, sizeof(LENTYPE));
    char buf[65536]; // hack!
    i.read(buf,len);
    s = buf;
    return len + sizeof(LENTYPE);
  }

  void write(std::ostream& o, const std::string& s) const
  {
    LENTYPE len = s.size()+1;
    o.write((char*) &len, sizeof(LENTYPE));
    o.write((char*) s.c_str(), len);
  }
}; // LENTYPE

typedef StringSerialiser<unsigned char> StringUnsignedShortSerialiser;

#endif
