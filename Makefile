SUBMISSION_FILES=HistogramImageFrequency.png Random50SpeedSign.png NoPassing-LumaAugment.png NoPassing-LumaAugment2.png NoPassing-LumaAugmentCorrupt.png StopSign.png StopSignV2.png NoParkingLeft.png FrostHazard.png AlertSign.png writeup_template.md Traffic_Sign_Classifier.ipynb report.html

# Compile the project zip file
project.zip: $(SUBMISSION_FILES)
	zip project.zip $(SUBMISSION_FILES)


