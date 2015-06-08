// Python wrapper for PCRF

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
  : crf_model(m), config(conf), crf_applier(m,conf)
  {}
  
  /// Apply model to input string
  std::string apply_to(std::string input)
  {
    std::stringstream in_sstr(input);
    std::stringstream out_sstr;
    JSONOutputter json_outputter(out_sstr,false);

    json_outputter.prolog();
    crf_applier.apply_to(in_sstr,json_outputter,true);
    json_outputter.epilog();

    return out_sstr.str();
  }

  // Apply to UTF-8 text file
  std::string apply_to_text_file(std::string filename) 
  {
    std::ifstream in(filename.c_str());
    if (!in) return "";

    std::stringstream out_sstr;
    JSONOutputter json_outputter(out_sstr,false);

    json_outputter.prolog();
    crf_applier.apply_to(in,json_outputter,true);
    json_outputter.epilog();

    return out_sstr.str();
  }
  
  
private:
  const SimpleLinearCRFModel<ORDER>&  crf_model;
  const NERConfiguration&             config;
  CRFApplier<ORDER>                   crf_applier; 
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
    def("apply_to_text_file",&FirstOrderLCRFApplier::apply_to_text_file);    
}

