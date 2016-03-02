__author__ = 'HyNguyen'
import os



if __name__ == "__main__":

    groups = [80,160,240,320]
    for group in groups:

        model_path = "data/summarymds/summary_model/" + str(group)
        system_path = "data/summarymds/summary_system/"  + str(group)
        file_names = os.listdir(model_path)
        print file_names

        counter  = 0
        config_file = '<ROUGE_EVAL version="1.55">'
        for file_names in file_names:
            config_file += '\t<EVAL ID="' + str(counter) + '">\n'
            config_file += '\t\t<MODEL-ROOT>\n'
            config_file += '\t\t\t'+model_path+'\n'
            config_file += '\t\t</MODEL-ROOT>\n'
            config_file += '\t\t<PEER-ROOT>\n'
            config_file += '\t\t\t'+system_path+'\n'
            config_file += '\t\t</PEER-ROOT>\n'
            config_file += '\t\t<INPUT-FORMAT TYPE="SPL">\n'
            config_file += '\t\t</INPUT-FORMAT>\n'
            config_file += '\t\t<PEERS>\n'
            config_file += '\t\t\t<P ID="1">'+file_names +'</P>\n'
            config_file += '\t\t</PEERS>\n'
            config_file += '\t\t<MODELS>\n'
            config_file += '\t\t\t<M ID="A">'+file_names+'</M>\n'
            config_file += '\t\t</MODELS>\n'
            config_file += '\t</EVAL>\n\n'
            counter +=1
        config_file += '</ROUGE_EVAL>\n'
        file_settings = open('data/summarymds/settings.'+str(group)+'.xml','w')
        file_settings.write(config_file)
        file_settings.close()