from giovanniScripts.persistance import save_author_file
from giovanniScripts.utils import abort_clean, create_dir
from os import listdir
from os.path import isfile, join
from time import time

def save_xml(inputDic, inputPath, outputPath, verbosity_level=3):
    '''

    :param inputDic: input dictionary in the form { 'user_id_1' : {
                                                        'text' : 'female' or 'male'
                                                        'image' : 'female' or 'male'
                                                        'comb' : 'female' or 'male'
                                                    },
                                                    ...
                                                    ...
                                                    ,user_id_N' : {
                                                        'text' : 'female' or 'male'
                                                        'image' : 'female' or 'male'
                                                        'comb' : 'female' or 'male'
                                                    }
                                                 }
    :param inputPath: path to the PAN18 dataset
    :param outputPath: path to the directory that will contain the ouput xml files for each
                       language according to PAN18 specifications
    :param verbosity_level: from 1 to 3
    :return:
    '''
    create_dir(outputPath)
    # ----------------------------------------------------------------------
    if verbosity_level:
        print('---------------------------------------')
        print("-----------Starting save_xml------------")
        print('---------------------------------------')
    # PAN 18 specifics
    for lang in ['ar','en','es']:
        input_dir = inputPath +"/"+ lang
        output_subdir_path = outputPath +"/"+ lang+"/"
        # ----------------------------------------------------------------------
        # Load the user_ids
        if verbosity_level:
            t0 = time()
            print("Starting files Listing ...")
        try:
            user_ids = [f[:-4] for f in listdir(input_dir + "/text") if (
                    isfile(join(input_dir + "/text", f)) and f[-4:] == ".xml")
                         ]
        except:
            abort_clean("Files listing --- failure",
                        "Maybe the directory specified is incorrect ?")
        if verbosity_level:
            print("Files found : " + str(len(user_ids)))
            print("Files listing --- success in %.3f seconds\n" % (time() - t0))

        if not (user_ids):
            abort_clean("Users loading failed")

        Authors = []
        for user_id in user_ids:
            auth = dict()
            auth['id'] = user_id
            auth['lang'] = lang
            auth['gender_txt'] = inputDic[user_id]['text']
            auth['gender_img'] = inputDic[user_id]['image']
            auth['gender_comb'] = inputDic[user_id]['comb']
            Authors.append(auth)
        #-----------------------------------------------------Save output XML
        create_dir(output_subdir_path)
        for auth in Authors:
            save_author_file(
                author=auth,
                output_dir=output_subdir_path,
                verbose=verbosity_level > 1
            )

'''
if __name__ == "__main__":
    inputPath = "pan18"
    outputPath = "output_txt_train"
    indic = dict()
    for lang in ['ar','en','es']:
        input_dir = inputPath +"/"+ lang
        user_ids = [f[:-4] for f in listdir(input_dir + "/text") if (
                isfile(join(input_dir + "/text", f)) and f[-4:] == ".xml")]
        for user_id in user_ids:
            indic[user_id] = dict()
            indic[user_id]['text'] = 'female'
            indic[user_id]['image'] = 'male'
            indic[user_id]['comb'] = 'female'

    save_xml(inputDic=indic, inputPath="pan18",outputPath="save_xml_output",verbosity_level=3)'''
