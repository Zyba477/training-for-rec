import os
import pandas as pd
import numpy as np
from mode import Mode


class Output:
    """
        result.txt
        - configuration of the model
        - classification report
        - accuracy / loss
        confusion_matrix.csv
        - confusion matrix
    """
    def __init__(self, report, program_mode):

        self.report = report
        self.mode = program_mode.get_mode()
        self.description = program_mode.__info_mode__()
        self.directory = program_mode.get_info_trained()


    def create_report_directory(self):
        self.__mkdir__("Output")
        
        self.__mkdir__(self.directory)
        
        self.__create_result_file__()
        self.__create_confusion_matrix_file__()
                 
    
    def __create_confusion_matrix_file__(self):

        df = pd.DataFrame(self.report['confusion matrix'])
        df.to_csv('confusion_matrix')


    def __create_result_file__(self):
        
        data = ['loss', 'accuracy', 'classification report']

        with open('result.txt', 'w') as f:

            f.write(self.description + '\n')
            f.write('\n')

            for param in data:

                line = str(self.report[param]) + '\n'

                if param == data[0]:
                    line = "Loss: {:.4}".format(self.report[param])
                elif param == data[1]:
                    line = "Accuracy: {:.2%}\n".format(self.report[param])

                f.write(str(line))
                f.write('\n')

            f.close()
    

    def __mkdir__(self, dirname):

        curr_dir = os.getcwd()
        path = os.path.join(curr_dir, dirname)

        if not os.path.exists(path):
            os.mkdir(path)

        os.chdir(path)