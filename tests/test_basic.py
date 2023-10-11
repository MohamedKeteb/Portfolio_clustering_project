import unittest
import yfinance as yf 
import pandas as pd 
from datetime import datetime
import os
import sys

# Obtenir le chemin du répertoire parent du répertoire contenant test_basic.py
current_directory = os.path.dirname(__file__)
project_directory = os.path.abspath(os.path.join(current_directory, ".."))

# Ajouter le chemin du répertoire 'code' au chemin de recherche Python
code_directory = os.path.join(project_directory, "code")
sys.path.append(code_directory)

# Maintenant, vous pouvez importer module1
import module1


class TestModule1Functions(unittest.TestCase):
    def test_function1(self):
        tickers_list = ['AAPL']
        result = module1.get_returns('2001-01-01', '2001-02-01', tickers_list)
        df = pd.DataFrame(yf.download(tickers_list, '2001-01-01','2001-02-01'))
        self.assertEqual(result, df)

if __name__ == '__main__':
    unittest.main()
