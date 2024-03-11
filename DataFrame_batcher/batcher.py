import pandas as pd

class BatchGenerator:
    def __init__(self, batch_size=50, df: pd.DataFrame = None) -> None:
        """
        Initialize the BatchGenerator class.

        Args:
            batch_size (int): The size of each batch.
            df (pd.DataFrame): The input DataFrame.
        """
        self.batch_size = batch_size
        self.batch_index = 0
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()

    def set_df(self, df: pd.DataFrame):
        """
        Set the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df.copy()

    def give_batch(self):
        """
        Get the next batch from the DataFrame.

        Returns:
            pd.DataFrame: The next batch of data.
        """
        if not isinstance(self.df, pd.DataFrame):
            print('Please provide a DataFrame.')
            return

        if self.batch_index >= len(self.df):
            print('All data has been fetched.')
            return None

        batch = self.df.iloc[self.batch_index:self.batch_index + self.batch_size].reset_index(drop=True)
        self.batch_index += self.batch_size

        return batch

    def __iter__(self):
        """
        Initialize the iterator.

        Returns:
            self: The BatchGenerator object.
        """
        return self

    def __next__(self):
        """
        Get the next batch from the DataFrame.

        Returns:
            pd.DataFrame: The next batch of data.

        """
        if not isinstance(self.df, pd.DataFrame):
            print('Please provide a DataFrame.')
            raise StopIteration

        if self.batch_index >= len(self.df):
            raise StopIteration

        batch = self.df.iloc[self.batch_index:self.batch_index + self.batch_size]
        self.batch_index += self.batch_size

        return batch