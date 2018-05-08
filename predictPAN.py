import numpy as np
import pandas as pd
import pickle

def parse_gender_dict(truthFilePath):
	with open(truthFilePath) as f:
		content = f.readlines()
		content = [x.strip() for x in content]
		
	genders = dict()
	# Female label is 0 ; Male label is 1
	for author_info in content:
		infos = author_info.split(':::')
		current_author_gender = None
		if(infos[1] == 'female'):
			current_author_gender = 0
		else:
			current_author_gender = 1
		genders[infos[0]] = current_author_gender
	
	return genders


from scipy.sparse import csr_matrix

def X_ToSparseMatrix_objectDetection(X_train, convertLabelToInt):
    columns = []
    rows = []
    values = []
    rowIndex = 0

    labelToIntCount = 0
    
    for observation in X_train:
        for objectLabel in observation['labels']:
            if objectLabel in convertLabelToInt:
                columns.append(convertLabelToInt[objectLabel])
                rows.append(rowIndex)
                values.append(observation['labels'][objectLabel])
        
        rowIndex += 1 # next observation item
    
    row  = np.array(rows)
    col  = np.array(columns)
    data = np.array(values)
    numberOfRows = len(X_train)
    numberOfColumns = len(convertLabelToInt)
    resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))
    
    return resultSparseMatrix, convertLabelToInt

def X_ToSparseMatrix_colorHistogram(X_train):	
    columns = []
    rows = []
    values = []
    rowIndex = 0

    convertLabelToInt = dict()
    color_histogram_flattened_length = len(X_train[0]['color_histogram'])

    for observation in X_train:
        observation_color_histogram = observation['color_histogram']
        color_histogram_index = 0
        while (color_histogram_index < len(observation_color_histogram)):
            columns.append(color_histogram_index)
            rows.append(rowIndex)
            values.append(observation_color_histogram[color_histogram_index])
            color_histogram_index += 1
        
        rowIndex += 1 # next observation item
    
    row  = np.array(rows)
    col  = np.array(columns)
    data = np.array(values)
    numberOfRows = len(X_train)
    numberOfColumns = color_histogram_flattened_length + len(convertLabelToInt)
    resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))
    
    return (resultSparseMatrix, convertLabelToInt)

def X_ToSparseMatrix_lbp(X_train):	
    columns = []
    rows = []
    values = []
    rowIndex = 0

    convertLabelToInt = dict()
    color_histogram_flattened_length = len(X_train[0]['local_binary_patterns'])

    for observation in X_train:
        observation_color_histogram = observation['local_binary_patterns']
        color_histogram_index = 0
        while (color_histogram_index < len(observation_color_histogram)):
            columns.append(color_histogram_index)
            rows.append(rowIndex)
            values.append(observation_color_histogram[color_histogram_index])
            color_histogram_index += 1
        
        rowIndex += 1 # next observation item
    
    row  = np.array(rows)
    col  = np.array(columns)
    data = np.array(values)
    numberOfRows = len(X_train)
    numberOfColumns = color_histogram_flattened_length + len(convertLabelToInt)
    resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))
    
    return (resultSparseMatrix, convertLabelToInt)
	
def X_ToSparseMatrix_faceRecognition(X_train, ids):	
	columns = []
	rows = []
	values = []
	rowIndex = 0

	convertLabelToInt = dict()
	labelToIntCount = 0

	for observation in X_train:
		if 'Female' in observation:
			if observation['Female'] > 0:
				columns.append(0)
				rows.append(rowIndex)
				values.append(observation['Female'])
		else:
			print('No Female attribute for ', ids[rowIndex])
			
		if 'Male' in observation:
			if observation['Male'] > 0:
				columns.append(1)
				rows.append(rowIndex)
				values.append(observation['Male'])
		else:
			print('No Male attribute for ', ids[rowIndex])
		
		rowIndex += 1 # next observation item

	row  = np.array(rows)
	col  = np.array(columns)
	data = np.array(values)
	numberOfRows = len(X_train)
	numberOfColumns = 2
	resultSparseMatrix = csr_matrix((data, (row, col)), shape=(numberOfRows, numberOfColumns))

	return (resultSparseMatrix, convertLabelToInt)
	
def createDataset(author_images, gender_dict):
	dataset = dict()
	for author in author_images:
		author_id = author
		author_gender = gender_dict[author_id]
		for imageIndex in author_images[author_id]:
			features = author_images[author_id][imageIndex]
			
			dataset[str(author_id + '.' + str(imageIndex))] = [features, author_gender]
	
	return dataset



def getXandYandIds(dataset):
	X = []
	y = []
	ids = []
	for author in dataset:
		X.append(dataset[author][0])
		y.append(dataset[author][1])
		ids.append(author)
	return X, y, ids
	
def predict(options):

	# ----------------------
	# -- IMAGE PREDICTION --
	
	
	# Object recognition features
	print('Extracting object recognition features')
	import feature_extractors.yolo_extractor as ye
	author_images_object_detection = ye.extractObjectDetectionFeatures(options['dataset_path'])
	#with open(options['features_path_yolo'] , "rb" ) as input_file:
	#	author_images_object_detection = pickle.load(input_file)
	
	# Face recognition features
	print('Extracting face recognition features')
	import feature_extractors.faceRecognition_extractor as fre
	author_images_face_recognition = fre.extractFaceRecognitionFeatures(options['dataset_path'])
	#with open(options['features_path_face_recognition'] , "rb" ) as input_file:
	#	author_images_face_recognition = pickle.load(input_file)
	
	# Global features
	print('Extracting global features')
	import feature_extractors.globalFeatures_extractor as gfe
	author_images_global_features = gfe.extractGlobalFeatures(options['dataset_path'])
	#with open(options['features_path_global_features'] , "rb" ) as input_file:
	#	author_images_global_features = pickle.load(input_file)
	
	
	# Concatening all features in a big author_images
	print('Concatening features')
	for author in author_images_object_detection:
		if author in author_images_global_features:
			for image_index in author_images_object_detection[author]:
				if image_index in author_images_global_features[author]:
					author_images_object_detection[author][image_index].update(author_images_global_features[author][image_index])
		if author in author_images_face_recognition:
			for image_index in author_images_object_detection[author]:
				if image_index in author_images_face_recognition[author]:
					author_images_object_detection[author][image_index].update(author_images_face_recognition[author][image_index])
	author_images = author_images_object_detection
	
	# Loading the truth file, for each language
	print('Loading the truth file, for each language')
	gender_dict = parse_gender_dict(options['dataset_path'] + '/ar/ar.txt')
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/en/en.txt'))
	gender_dict.update(parse_gender_dict(options['dataset_path'] + '/es/es.txt'))
	
	# Creating a dict with an incremental id as key and the array [features, author_gender] as value, for each image
	print('Creating dataset')
	dataset = createDataset(author_images, gender_dict)
	
	# Getting X and Y vectors from the dataset
	X, y, ids = getXandYandIds(dataset)
	
	# Converting X to sparse matrix, for each type of feature
	convertLabelToInt = None
	with open(options['convertLabelToInt_path'] , "rb" ) as input_file:
		convertLabelToInt = pickle.load(input_file)
	print('Converting X to sparse matrix object detect')
	X_objectDetection, convertLabelToInt = X_ToSparseMatrix_objectDetection(X, convertLabelToInt)
	print('Converting X to sparse matrix CH')
	X_colorHistogram, convertLabelToInt = X_ToSparseMatrix_colorHistogram(X)
	print('Converting X to sparse matrix lbp')
	X_lbp, convertLabelToInt = X_ToSparseMatrix_lbp(X)
	print('Converting X to sparse matrix face recognition')
	X_faceRecognition, convertLabelToInt = X_ToSparseMatrix_faceRecognition(X, ids)
	
	# Loading classifiers
	print('Loading object detection classifier')
	objectDetectionClf = None
	with open(options['clf_path_object_detection'] , "rb" ) as input_file:
		objectDetectionClf = pickle.load(input_file)
	
	print('Loading face detection classifier')
	faceRecognitionClf = None
	with open(options['clf_path_face_recognition'] , "rb" ) as input_file:
		faceRecognitionClf = pickle.load(input_file)
	
	print('Loading lbp classifier')
	lbpClf = None
	with open(options['clf_path_lbp'] , "rb" ) as input_file:
		lbpClf = pickle.load(input_file)
	
	print('Loading color histogram classifier')
	colorHistogramClf = None
	with open(options['clf_path_color_histogram'] , "rb" ) as input_file:
		colorHistogramClf = pickle.load(input_file)
	
	# Making the predictions for each classifier
	print('Getting prediction for face recognition')
	pred_face_recognition = faceRecognitionClf.predict_proba(X_faceRecognition)
	print('Getting prediction for color histogram')
	pred_color_histogram = colorHistogramClf.predict_proba(X_colorHistogram)
	print('Getting prediction for lbp')
	pred_lbp = lbpClf.predict_proba(X_lbp)
	print('Getting prediction for object recognition')
	pred_object_detection = objectDetectionClf.predict_proba(X_objectDetection)
	
	# Building the meta image input data from the output of the classifiers for each entry
	print('Building the meta image input data')
	input_meta_image = []
	i=0
	while i < len(pred_face_recognition):
		current_input_entry = [pred_face_recognition[i][0], pred_face_recognition[i][1],
								pred_color_histogram[i][0], pred_color_histogram[i][1],
								pred_lbp[i][0], pred_lbp[i][1],
								pred_object_detection[i][0], pred_object_detection[i][1]]
		input_meta_image.append(current_input_entry)
		i+=1
	
	# Getting the meta image prediction for each author
	print('Getting the meta image prediction for each author')
	metaImageClf = None
	with open(options['clf_path_meta_image'] , "rb" ) as input_file:
		metaImageClf = pickle.load(input_file)
	
	pred_meta_image = metaImageClf.predict_proba(input_meta_image)
	
	# Aggregating the results
	print('Aggregating the results')
	authorId_location_map = dict()
	i=0
	for id in ids:
		author_id = id.split('.')[0]
		if author_id not in authorId_location_map:
			authorId_location_map[author_id] = []
		authorId_location_map[author_id].append(i)
		i += 1
			

	import numpy
	
	aggregated_predictions = []
	yAggregated = []
	aggregatedIds = []
	for author in authorId_location_map:
		aggregated_predictions_current_author = []
		numberOfImagesForAuthor = 0
		for location in authorId_location_map[author]:
			aggregated_predictions_current_author.append(pred_meta_image[location][0])
			aggregated_predictions_current_author.append(pred_meta_image[location][1])
			numberOfImagesForAuthor += 1
			
		while numberOfImagesForAuthor < 10:
			aggregated_predictions_current_author.append(0.5)
			aggregated_predictions_current_author.append(0.5)
			numberOfImagesForAuthor += 1
			
		aggregated_predictions.append(aggregated_predictions_current_author)
		yAggregated.append(y[authorId_location_map[author][0]])
		aggregatedIds.append(author)
	
	aggregated_predictions = numpy.array(aggregated_predictions)
	
	# Getting the predictions of the image aggregation classifier
	print('Getting the predictions of the image aggregation classifier')
	aggregationClf = None
	with open(options['clf_path_aggregation'] , "rb" ) as input_file:
		aggregationClf = pickle.load(input_file)
	
	pred_aggregation_image = aggregationClf.predict_proba(aggregated_predictions)
	
	# Getting the plain prediction for image
	print('Getting the plain prediction for image')
	pred_image = dict()
	i=0
	while i < len(pred_aggregation_image):
		if pred_aggregation_image[i][0] > 0.5:
			pred_image[aggregatedIds[i]] = 'female'
		else:
			pred_image[aggregatedIds[i]] = 'male'
		i += 1
	
	
	# ---------------------
	# -- TEXT PREDICTION --
	
	# Constructing the input dictionnary for text prediction
	# 'ar':[arUser0, .. , arUserN],
    #                    'en':[enUser0, .. , enUserN]
    #                    'es':[esUser0, .. , esUserN]}
	
	gender_dict_ar = parse_gender_dict(options['dataset_path'] + '/ar/ar.txt')
	gender_dict_en = parse_gender_dict(options['dataset_path'] + '/en/en.txt')
	gender_dict_es = parse_gender_dict(options['dataset_path'] + '/es/es.txt')
	
	authorId_dict = dict()
	authorId_dict['ar'] = []
	authorId_dict['en'] = []
	authorId_dict['es'] = []
	for author_id in gender_dict_ar:
		authorId_dict['ar'].append(author_id)
	for author_id in gender_dict_en:
		authorId_dict['en'].append(author_id)
	for author_id in gender_dict_es:
		authorId_dict['es'].append(author_id)
		
	# Getting the text prediction
	import text_prediction
	print('Getting the text prediction')
	text_predictions = text_prediction.predict(options['dataset_path'], authorId_dict, options['text_clf_path'])
	
	# Getting the plain prediction for text
	print('Getting the plain prediction for text')
	pred_text = dict()
	for author in text_predictions:
		if(float(text_predictions[author][0]) > 0.5):
			pred_text[author] = 'female'
		else:
			pred_text[author] = 'male'
	
	
	
	# -------------------------
	# -- COMBINED PREDICTION --
	
	# Building combined prediction input
	print('Building combined prediction input')
	
	combined_prediction_input = []
	i=0
	yAggregatedFinal = []
	while i < len(aggregatedIds):
		current_combined_prediction = []
		current_combined_prediction.append(float(pred_aggregation_image[i][0]))
		current_combined_prediction.append(float(pred_aggregation_image[i][1]))
		# Getting the prediction for text, corresponding to the current author
		current_author_text_prediction = text_predictions[aggregatedIds[i]]
		current_combined_prediction.append(float(current_author_text_prediction[0]))
		current_combined_prediction.append(float(current_author_text_prediction[1]))
		
		combined_prediction_input.append(current_combined_prediction)
		yAggregatedFinal.append(gender_dict[aggregatedIds[i]])
		i+=1
	
	combined_prediction_input = numpy.array(combined_prediction_input)
	yAggregatedFinal = numpy.array(yAggregatedFinal)
	
	# Loading combined text-image classifier
	print('Loading combined text-image classifier')
	combinedTextImageClf = None
	with open(options['clf_path_combined_clf'] , "rb" ) as input_file:
		combinedTextImageClf = pickle.load(input_file)
	
	# Getting the final text-image combined prediction
	print('Getting the final text-image combined prediction')
	combined_prediction = combinedTextImageClf.predict(combined_prediction_input)
	
	# Constructing the final text-image combined prediction dict
	print('Constructing the final text-image combined prediction dict')
	pred_combined = dict()
	i=0
	while i < len(combined_prediction):
		if combined_prediction[i] == 0:
			pred_combined[aggregatedIds[i]] = 'female'
		elif combined_prediction[i] == 1:
			pred_combined[aggregatedIds[i]] = 'male'
		else:
			print('Error in constructing the final text-image combined prediction dict')
		i += 1
	
	
	
	# -------------------------
	# -- SAVING RESULTS --
	
	# Merging the predictions in one dictionary
	print('Merging the predictions in one dictionary')
	dict_predict_merged = dict()
	for author in pred_combined:
		dict_predict_merged[author] = { 'text': pred_text[author], 'image': pred_image[author], 'comb': pred_combined[author] }
	
	# Saving results as xml file
	print('Saving results as xml files')
	import save_xml
	print(len(dict_predict_merged))
	save_xml.save_xml(inputDic=dict_predict_merged, inputPath=options['dataset_path'], outputPath=options['output_save'], verbosity_level=1)


if __name__ == "__main__":
	options = {
		'features_path_yolo': "feature_extractors/extracted-features/author_images_yolo-all.p",
		'features_path_global_features': "feature_extractors/extracted-features/author_images_global_features-all.p",
		'features_path_face_recognition': "feature_extractors/extracted-features/author_images_face_recognition-all.p",
		'clf_path_object_detection': "./trained-classifiers/object-detection-classifier.p",
		'clf_path_lbp': "./trained-classifiers/lbp-classifier.p",
		'clf_path_color_histogram': "./trained-classifiers/color-histogram-classifier.p",
		'clf_path_face_recognition': "./trained-classifiers/face-recognition-classifier.p",
		'clf_path_meta_image': "./trained-classifiers/meta-image-classifier.p",
		'clf_path_aggregation': "./trained-classifiers/aggregation-classifier.p",
		'clf_path_combined_clf': "./trained-classifiers/final-text-image-classifier.p",
		'convertLabelToInt_path': './labels_object_detection/convertLabelToInt.p',
		'dataset_path': "./PAN dataset/pan18-author-profiling-training-2018-02-27",
		'text_clf_path': "./output_txt_train",
		'split_path': "../output/splitting-image-text"
	}
	
	# LOADING PARSER
	import argparse
	
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset", help="Path to the whole dataset")
	parser.add_argument("--output", help="Path to save the result of the prediction as xml files")

	args = parser.parse_args()
	options['dataset_path'] = args.dataset
	options['output_save'] = args.output
		
	
	predict(options)