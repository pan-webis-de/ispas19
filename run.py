import sys

import model
import rw_operations

input_dataset = sys.argv[2]
output_dir = sys.argv[4]

if not str(input_dataset).endswith('/'):
    input_dataset += '/'
if not str(output_dir).endswith('/'):
    output_dir += '/'

print('Run with params\n input_dataset:', input_dataset, '\n', 'output_dir:', output_dir)


language = 'en'
print('Train and predict for language: ', language)
classifier = model.NuSVClassifier(language, input_dataset)
classifier.fit()

predictions = classifier.predict()
rw_operations.save_predictions(predictions, language, output_dir)


language = 'es'
print('Train and predict for language: ', language)
classifier = model.NuSVClassifier(language, input_dataset)
classifier.fit()

predictions = classifier.predict()
rw_operations.save_predictions(predictions, language, output_dir)
