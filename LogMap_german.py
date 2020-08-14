import pickle, time, subprocess, os

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

prefix = "/data/Vivek/IBM/IBM-Internship/"
for ont_pair in ontologies_in_alignment_new:
		print ("Doing {}...".format(ont_pair))
		t = time.time()
		a, b, c = prefix + ont_pair[0], prefix + ont_pair[1], ont_pair[0].split("/")[-1].rsplit(".",1)[0].replace(".", "_").lower() + "-" + ont_pair[1].split("/")[-1].rsplit(".",1)[0].replace(".", "_").lower()
		os.mkdir(c)
		java_command = "java -jar logmap-matcher/target/logmap-matcher-4.0.jar MATCHER file:" +  a + \
										 " file:" + b + " " + "/data/Vivek/IBM/IBM-Internship/" + c + "/ false"
		process = subprocess.Popen(java_command.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()
		print ("Took {} seconds for {}".format(time.time()-t, ont_pair))
