// Python wrapper for PCRF

#include <string>
#include <fstream>
#include <sstream>

#include <boost/python.hpp>

#include <NERConfiguration.hpp>
#include <SimpleLinearCRFModel.hpp>
#include <CRFApplier.hpp>
#include <NEROutputters.hpp>

/// CRF applier
template<unsigned ORDER>
class LCRFApplier
{
public:
  LCRFApplier(const SimpleLinearCRFModel<ORDER>& m, const NERConfiguration& conf)
  : crf_model(m), config(conf), crf_applier(m,conf), out_sstr(new std::stringstream)
  {
    json_outputter = new JSONOutputter(*out_sstr,false);
    tsv_outputter = new NEROneWordPerLineOutputter(*out_sstr);
    current_outputter = tsv_outputter;
  }
  
  ~LCRFApplier()
  {
    delete json_outputter;
    delete tsv_outputter;
  }
  
  /// Apply model to input string
  std::string apply_to(std::string input)
  {
    std::stringstream in_sstr(input);
    out_sstr->clear();
    out_sstr->str("");
    
    current_outputter->prolog();
    crf_applier.apply_to(in_sstr,*current_outputter,true);
    current_outputter->epilog();

    return out_sstr->str();
  }

  // Apply to UTF-8 text file
  std::string apply_to_text_file(std::string filename) 
  {
    std::ifstream in(filename.c_str());
    if (!in) return "";

    out_sstr->clear();
    out_sstr->str("");
    current_outputter->prolog();
    crf_applier.apply_to(in,*current_outputter,true);
    current_outputter->epilog();

    return out_sstr->str();
  }
  
  void set_output_mode(std::string mode)
  {
    if (mode == "json") current_outputter = json_outputter;
    else if (mode == "tsv") current_outputter = tsv_outputter;
    else std::cerr << "PCRF: Error: Unknown output mode" << std::endl;
  }
  
    
private:
  const SimpleLinearCRFModel<ORDER>&  crf_model;
  const NERConfiguration&             config;
  CRFApplier<ORDER>                   crf_applier;
  std::shared_ptr<std::stringstream>  out_sstr;
  JSONOutputter*                      json_outputter;
  NEROneWordPerLineOutputter*         tsv_outputter;
  NEROutputterBase*                   current_outputter;
}; // LCRFApplier


typedef LCRFApplier<1>          FirstOrderLCRFApplier;
typedef SimpleLinearCRFModel<1> SimpleLinearCRFFirstOrderModel;



// Define the PCRF Python module  
BOOST_PYTHON_MODULE(pcrf_python)
{
  using namespace boost::python;

  class_<NERConfiguration>("NERConfiguration", init<std::string>()).
    def("set_running_text_input", &NERConfiguration::set_running_text_input);

  class_<SimpleLinearCRFFirstOrderModel>("SimpleLinearCRFFirstOrderModel",
                                         init<std::string>());

  class_<FirstOrderLCRFApplier>("FirstOrderLCRFApplier",
                                init<const SimpleLinearCRFFirstOrderModel&, const NERConfiguration&>()).
    def("apply_to",&FirstOrderLCRFApplier::apply_to).
    def("apply_to_text_file",&FirstOrderLCRFApplier::apply_to_text_file).
    def("set_output_mode",&FirstOrderLCRFApplier::set_output_mode);    
}

