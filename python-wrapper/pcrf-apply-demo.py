import pcrf_python
import json

# Load config file
config = pcrf_python.NERConfiguration("../rtr/ner.cfg")

config.set_running_text_input(True)

# Load model
model = pcrf_python.SimpleLinearCRFFirstOrderModel("../rtr/rtr.model")

# Construct applier on the basis of the CRF model and the configuration
crf_applier = pcrf_python.FirstOrderLCRFApplier(model,config)

# Apply to input string
#utf8_string = "Merkel and Obama met at the G7 summit at Schloss Elmau near Garmisch-Partenkirchen."

# Apply to file
json_string = crf_applier.apply_to_text_file("cl-final.txt")
print json_string

# Convert to json object and pretty print
d = json.loads(json_string)
print json.dumps(d,indent=2)

