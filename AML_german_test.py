import pickle, time, subprocess

ontologies_in_alignment = pickle.load(open("data_german_dataset_phrase.pkl", "rb"))[-1]
ontologies_in_alignment_new = \
[('german_datasets_copy/lebensmittel/Google.Lebensmittel.owl',
  'german_datasets_copy/lebensmittel/web.Lebensmittel.owl'),
 ('german_datasets_copy/freizeit/dmoz.Freizeit.owl',
  'german_datasets_copy/freizeit/Google.Freizeit.owl'),
 ('german_datasets_copy/webdirectory/dmoz.owl',
  'german_datasets_copy/webdirectory/google.owl'),
 ('german_datasets_copy/webdirectory/dmoz.owl',
  'german_datasets_copy/webdirectory/web.owl'),
 ('german_datasets_copy/webdirectory/dmoz.owl',
  'german_datasets_copy/webdirectory/yahoo.small.owl'),
 ('german_datasets_copy/webdirectory/google.owl',
  'german_datasets_copy/webdirectory/web.owl'),
 ('german_datasets_copy/webdirectory/google.owl',
  'german_datasets_copy/webdirectory/yahoo.small.owl'),
 ('german_datasets_copy/webdirectory/web.owl',
  'german_datasets_copy/webdirectory/yahoo.small.owl')]

for ont_pair in ontologies_in_alignment_new:
    print ("Doing {}...".format(ont_pair))
    t = time.time()
    a, b, c = ont_pair[0], ont_pair[1], ont_pair[0].split("/")[-1].rsplit(".",1)[0].replace(".", "_").lower() + "-" + ont_pair[1].split("/")[-1].rsplit(".",1)[0].replace(".", "_").lower()
    java_command = "java -jar AML_v3.1/AgreementMakerLight.jar -s " + a + " -t " + b + " -o AML-test-results/" + c + ".rdf -a"
    process = subprocess.Popen(java_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print ("Took {} seconds for {}".format(time.time()-t, ont_pair))
