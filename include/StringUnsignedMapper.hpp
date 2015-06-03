#ifndef __STRINGUNSIGNEDMAPPER_HPP__
#define __STRINGUNSIGNEDMAPPER_HPP__

#include <boost/unordered_map.hpp>
#include <string>
#include <vector>


class StringUnsignedMapper
{
private:
  typedef boost::unordered_map<std::string,unsigned>  StringToUnsignedMap;
  typedef StringToUnsignedMap::const_iterator         StringToUnsignedIter;
  typedef std::vector<StringToUnsignedIter>           UnsignedToStringIndexMap;

public:
  typedef StringToUnsignedMap::const_iterator         const_iterator;

public:
  StringUnsignedMapper() : total_string_len(0)
  {
    id_string_map.reserve(1000);
  }

  void set_expected_size(size_t n)
  {
    id_string_map.resize(n);
  }

  bool add_pair(const std::string& s, unsigned id)
  {
    std::pair<StringToUnsignedMap::iterator,bool> p = string_id_map.insert(std::make_pair(s,id));
    if (p.first != string_id_map.end()) {
      if (id >= id_string_map.size()) {
        id_string_map.resize(unsigned((id*1.25)+10));
      }
      id_string_map[id] = p.first;
      total_string_len += (s.size()+1);
      return true;
    }
    else return false;
  }

  inline unsigned get_id(const std::string& s) const
  {
    StringToUnsignedMap::const_iterator f = string_id_map.find(s);
    return (f != string_id_map.end()) ? f->second : unsigned(-1);
  }

  const std::string& get_string(unsigned id) const
  {
    static std::string no_string("");
    return (id < id_string_map.size()) ? id_string_map[id]->first : no_string;
  }

  unsigned size() const
  {
    return string_id_map.size();
  }

  unsigned total_string_length() const
  {
    return total_string_len;
  }

  void compress()
  {
    UnsignedToStringIndexMap(id_string_map).swap(id_string_map);
  }

  void clear()
  {
    string_id_map.clear();
    UnsignedToStringIndexMap().swap(id_string_map);
    total_string_len = 0;
  }

  const_iterator begin() const
  {
    return string_id_map.begin();
  }

  const_iterator end() const
  {
    return string_id_map.end();
  }

  void print(std::ostream& out, std::string pref, std::string sep) const
  {
    for (unsigned i = 0; i < string_id_map.size(); ++i) {
      out << pref << i << sep << id_string_map[i]->first << std::endl;
    }
  }

  /// Reads the mapper from a binary file stream
  bool read(std::ifstream& in)
  {
    unsigned num_strings = 0;
    in.read((char*)&num_strings,sizeof(num_strings));

    if (num_strings == 0) {
      std::cerr << "Error (StringUnsignedMapper::read()): No strings found\n";
      return false;
    }

    in.read((char*)&total_string_len,sizeof(total_string_len));

    char* buf = new (std::nothrow) char[total_string_len];
    if (buf == 0) {
      std::cerr << "Error (StringUnsignedMapper::read()): Unable to allocate string buffer\n";
      return false;
    }

    std::vector<unsigned> ids(num_strings);
    set_expected_size(num_strings);
    in.read(buf,total_string_len);
    in.read((char*)&ids[0],ids.size()*sizeof(unsigned));
    
    char* p_buf = buf;
    for (unsigned i = 0; i < num_strings; ++i) {
      std::string s = std::string(p_buf);
      add_pair(s,ids[i]);
      p_buf += s.size()+1;
      //std::cout << "Adding " << label << std::endl;
    }

    delete[] buf;
    return true;
  }

  /// Writes the mapper to a binary file stream
  bool write(std::ofstream& out) const
  {
    char* buf = new char[total_string_len];
    if (buf == 0) {
      std::cerr << "StringUnsignedMapper::write: Error: Unable to allocate string buffer\n";
      return false;
    }

    unsigned num_strings = string_id_map.size();

    out.write((char*)&num_strings,sizeof(num_strings));
    out.write((char*)&total_string_len,sizeof(total_string_len));

    char* p_buf = buf;
    std::vector<unsigned> ids;
    ids.reserve(string_id_map.size());
    for (auto id = string_id_map.begin(); id != string_id_map.end(); ++id) {
      strcpy(p_buf,id->first.c_str());
      p_buf += id->first.size()+1;
      ids.push_back(id->second);
    }

    out.write(buf,total_string_len);
    out.write((char*)&ids[0],ids.size()*sizeof(unsigned));

    delete[] buf;
    return true;
  }

private:
  StringToUnsignedMap       string_id_map;
  UnsignedToStringIndexMap  id_string_map;
  unsigned                  total_string_len;
}; // StringUnsignedMapper

#endif