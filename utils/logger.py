import csv

class ExperimentLogger():

    def __init__(self, filename, fieldnames):        
        self.file = open(filename, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, data):
        self.writer.writerow(data)
        self.file.flush()

    def close(self):
        self.file.close()