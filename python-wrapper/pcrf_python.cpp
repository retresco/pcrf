// Python wrapper for PCRF

#include <string>
#include <fstream>
#include <sstream>

#include <boost/python.hpp>

#include <CRFConfiguration.hpp>
#include <SimpleLinearCRFModel.hpp>
#include <CRFApplier.hpp>
#include <NEROutputters.hpp>

/// Encapsulates all the necessary parts for applying a CRF model to some input
/// sequence within a single class
template<unsigned ORDER>
class LCRFApplier
{
public:
  /** 
    @brief Constructor
    @param m the CRF model
    @param conf the CRF configuration used during application
  */
  LCRFApplier(const SimpleLinearCRFModel<ORDER>& m, const CRFConfiguration& conf)
  : crf_model(m), config(conf), crf_applier(m,conf), out_sstr(new std::stringstream)
  {
    // Dynamically create two different outputters
    json_outputter = new JSONOutputter(*out_sstr,false);
    tsv_outputter = new NEROneWordPerLineOutputter(*out_sstr);
    // Set the default to tsv (tab-separated values)
    current_outputter = tsv_outputter;
  }
  
  /// Destructor
  ~LCRFApplier()
  {
    delete json_outputter;
    delete tsv_outputter;
  }
  
  /// Apply model to UTF-8-encoded input string
  std::string apply_to(std::string input)
  {
    std::stringstream in_sstr(input);
    out_sstr->clear();
    
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
    current_outputter->prolog();
    crf_applier.apply_to(in,*current_outputter,true);
    current_outputter->epilog();

    return out_sstr->str();
  }
  
  /// Determines the output mode (either json or tsv)
  void set_output_mode(std::string mode)
  {
    if (mode == "json") {
      json_outputter->reset();
      current_outputter = json_outputter;
    }
    else if (mode == "tsv") {
      tsv_outputter->reset();
      current_outputter = tsv_outputter;
    }
    else std::cerr << "PCRF: Error: Unknown output mode" << std::endl;
  }
  
  /// Reset the applier to a neutral state
  void reset()
  {
    out_sstr->str("");
    current_outputter->reset();
    crf_applier.reset();
  }
    
private: // Variables
  const SimpleLinearCRFModel<ORDER>&  crf_model;        ///< The CRF model
  const CRFConfiguration&             config;           ///< The underlying configuration
  CRFApplier<ORDER>                   crf_applier;      ///< The actual applier
  std::shared_ptr<std::stringstream>  out_sstr;         ///< All the output goes here
  JSONOutputter*                      json_outputter;   ///< Outputter for JSON strings
  NEROneWordPerLineOutputter*         tsv_outputter;
  NEROutputterBase*                   current_outputter;
}; // LCRFApplier


typedef LCRFApplier<1>            FirstOrderLCRFApplier;
typedef SimpleLinearCRFModel<1>   SimpleLinearCRFFirstOrderModel;


/// Define the PCRF Python module  
BOOST_PYTHON_MODULE(pcrf_python)
{
  using namespace boost::python;

  class_<CRFConfiguration>("CRFConfiguration", init<std::string>()).
    def("set_running_text_input", &CRFConfiguration::set_running_text_input);

  class_<SimpleLinearCRFFirstOrderModel>("SimpleLinearCRFFirstOrderModel",
                                         init<std::string>());

  class_<FirstOrderLCRFApplier>("FirstOrderLCRFApplier",
                                init<const SimpleLinearCRFFirstOrderModel&, const CRFConfiguration&>()).
    def("apply_to",&FirstOrderLCRFApplier::apply_to).
    def("apply_to_text_file",&FirstOrderLCRFApplier::apply_to_text_file).
    def("set_output_mode",&FirstOrderLCRFApplier::set_output_mode).
    def("reset",&FirstOrderLCRFApplier::reset)
  ;    
}
