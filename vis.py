import pandas
from IPython.display import display
import matplotlib.pyplot as plt

class Vis:

    def __init__(self, type, data):
        self.type = type    # instance variable unique to each instance
        self.data = data
        if (self.type == "summary"):
            self.vis_summary(self.data)
        else:
            raise Exception("Invalid visualization type")

    def vis_summary(self, data):
        fig1, ax1 = plt.subplots()
        data.boxplot(ax=ax1)
        ax1.set_title('Ratings Summary')
        plt.show()