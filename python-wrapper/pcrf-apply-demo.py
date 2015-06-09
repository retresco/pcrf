import pcrf_python
import json

# Load config file
config = pcrf_python.NERConfiguration("../demo/chunk.cfg")

config.set_running_text_input(True)

# Load model
model = pcrf_python.SimpleLinearCRFFirstOrderModel("../demo/chunker.model")

# Construct applier on the basis of the CRF model and the configuration
crf_applier = pcrf_python.FirstOrderLCRFApplier(model,config)
# Set the output mode to "tab-separated" (the other possibility is "json",
# but this currently works only for NER tasks)
crf_applier.set_output_mode("tsv")

# Apply to input string
utf8_string = "Merkel and Obama met at the G7 summit at Schloss Elmau near Garmisch-Partenkirchen."
out_string = crf_applier.apply_to(utf8_string)
print out_string

# Apply to file
out_string = crf_applier.apply_to_text_file("cl-final.txt")
print out_string

# Convert to json object and pretty print
#d = json.loads(out_string)
#print json.dumps(d,indent=2)

