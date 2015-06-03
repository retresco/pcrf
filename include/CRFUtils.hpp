
template<unsigned ORDER>
void model_info(const SimpleLinearCRFModel<ORDER>& crf_model)
{
  std::cerr << "============================================\n";
  std::cerr << "Model information\n";
  std::cerr << "============================================\n";
  std::cerr << "Order:         " << crf_model.model_order() << "\n";
  std::cerr << "# labels:      " << crf_model.labels_count() << "\n";
  std::cerr << "# states:      " << crf_model.states_count() << "\n";
  std::cerr << "# transitions: " << crf_model.transitions_count() << "\n";
  std::cerr << "# features:    " << crf_model.features_count() << "\n";
  std::cerr << "# attributes:  " << crf_model.attributes_count() << "\n";
  std::cerr << "# parameters:  " << crf_model.parameters_count();
  const ParameterVector& p = crf_model.get_parameters();
  unsigned nn = 0;
  for (unsigned i = 0; i < p.size(); ++i) {
    if (p[i] != Weight(0.0)) ++nn;
  }
  std::cerr << " (non-null: " << nn << ")\n";
  std::cerr << "============================================\n";
}

