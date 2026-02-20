import pandas as pd

class CleanData():
    """Minimal data cleaner for already well-structured input files.
    Applies only lightweight column/index formatting adjustments."""

    def __innit__(self,dfs):
        self.dfs = dfs
    
    def clean_data(self):

        for df in self.dfs:
            df = df.droplevel(1,axis=1)
            df.columns.name = None