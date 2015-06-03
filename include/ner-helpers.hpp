void load_binary_ne_list(const std::string& fn, NERFeatureExtractor& ner_fe)
{
  std::ifstream list_in(fn.c_str(),std::ios::binary);
  if (list_in) {
    std::cerr << " " << fn;
    ner_fe.add_ne_list(list_in);
  }
  else std::cerr << "\nError: crf-test: Unable to open NE list '" << fn << "'" << std::endl;
}


void load_name_list(const std::string& fn, NERFeatureExtractor& ner_fe)
{
  std::ifstream list_in(fn.c_str());
  if (list_in) {
    std::cerr << " " << fn;
    ner_fe.add_ne_list(list_in);
  }
  else std::cerr << "\nError: Unable to open named entity list '" << fn << "'" << std::endl;
}

void load_context_list(const char* fn, NERFeatureExtractor& ner_fe, bool left)
{
  std::ifstream list_in(fn);
  if (list_in) {
    std::cerr << " " << fn;
    if (left) ner_fe.add_left_context_list(list_in);
    else ner_fe.add_right_context_list(list_in);
  }
  else std::cerr << "\nError: crf-test: Unable to open context list '" << fn << "'" << std::endl;
}

void load_binary_context_list(const char* fn, NERFeatureExtractor& ner_fe, bool left)
{
  std::ifstream list_in(fn,std::ios::binary);
  if (list_in) {
    std::cerr << " " << fn;
    if (left) ner_fe.add_left_context_list(list_in);
    else ner_fe.add_right_context_list(list_in);
  }
  else std::cerr << "\nError: crf-test: Unable to open context list '" << fn << "'" << std::endl;
}

void load_binary_person_names_list(const char* fn, NERFeatureExtractor& ner_fe)
{
  std::ifstream list_in(fn,std::ios::binary);
  if (list_in) {
    std::cerr << " " << fn;
      ner_fe.add_person_names_list(list_in);
  }
  else std::cerr << "\nError: crf-test: Unable to open person names list '" << fn << "'" << std::endl;
}

void load_regex_list(const char* fn, NERFeatureExtractor& ner_fe)
{
  std::ifstream list_in(fn);
  if (list_in) {
    std::cerr << " " << fn;
    ner_fe.add_word_regex_list(list_in);
  }
  else std::cerr << "\nError: crf-test: Unable to open Regex list '" << fn << "'" << std::endl;
}

